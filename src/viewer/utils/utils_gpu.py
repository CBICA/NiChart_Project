# gpu_selector_widget.py
# Run with: streamlit run gpu_selector_widget.py
# Or import gpu_selector_widget.render(...) inside your app.

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import platform
import re
import shutil
import subprocess
from typing import List, Dict, Any, Optional

def _run(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return None

def _parse_nvidia_smi() -> List[Dict[str, Any]]:
    """
    Use nvidia-smi to get per-GPU info. Returns empty list if not available.
    """
    base = []
    if shutil.which("nvidia-smi") is None:
        return base

    # CUDA version & driver from banner
    banner = _run(["nvidia-smi"])
    cuda_ver = None
    driver_ver_from_banner = None
    if banner:
        m = re.search(r"CUDA Version:\s*([\d.]+)", banner)
        if m: cuda_ver = m.group(1)
        m = re.search(r"Driver Version:\s*([\d.]+)", banner)
        if m: driver_ver_from_banner = m.group(1)

    # Query per-GPU fields
    fields = [
        "name","uuid","pci.bus_id",
        "memory.total","memory.free",
        "driver_version","vbios_version",
        "temperature.gpu","utilization.gpu",
        "compute_mode"
    ]
    query = ",".join(fields)
    out = _run([
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits"
    ])
    if not out:
        return base

    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        rec = { "vendor": "NVIDIA" }
        for key, val in zip(fields, parts):
            k = key.replace(".", "_")
            rec[k] = val

        # Normalize memory & numeric fields where possible
        for k in ("memory_total","memory_free","temperature_gpu","utilization_gpu"):
            if k in rec:
                try:
                    rec[k] = int(rec[k])
                except Exception:
                    pass

        # Attach CUDA/driver
        rec["cuda_version"] = cuda_ver
        if not driver_ver_from_banner and rec.get("driver_version"):
            rec["driver_version"] = rec["driver_version"]
        else:
            rec["driver_version"] = driver_ver_from_banner or rec.get("driver_version")

        base.append(rec)

    return base

def _nvml_info() -> List[Dict[str, Any]]:
    """
    Try pynvml for richer NVIDIA details (including compute capability).
    """
    out = []
    try:
        import pynvml as N
        N.nvmlInit()
        count = N.nvmlDeviceGetCount()
        dr = N.nvmlSystemGetDriverVersion().decode() if isinstance(N.nvmlSystemGetDriverVersion(), bytes) else N.nvmlSystemGetDriverVersion()
        for i in range(count):
            h = N.nvmlDeviceGetHandleByIndex(i)
            name = N.nvmlDeviceGetName(h)
            if isinstance(name, bytes): name = name.decode()
            mem = N.nvmlDeviceGetMemoryInfo(h)
            uuid = N.nvmlDeviceGetUUID(h)
            if isinstance(uuid, bytes): uuid = uuid.decode()
            pci = N.nvmlDeviceGetPciInfo(h)
            bus_id = pci.busId
            if isinstance(bus_id, bytes): bus_id = bus_id.decode()
            rec = {
                "vendor": "NVIDIA",
                "name": name,
                "uuid": uuid,
                "pci_bus_id": bus_id,
                "memory_total": int(mem.total // (1024**2)),
                "memory_free": int(mem.free // (1024**2)),
                "driver_version": dr,
            }
            # Compute capability via nvml? Not directly—try CUDA via PyTorch later.
            out.append(rec)
        N.nvmlShutdown()
    except Exception:
        pass
    return out

def _rocm_info() -> List[Dict[str, Any]]:
    """
    Try rocm-smi for AMD GPUs. Returns per-GPU dicts if available.
    """
    out = []
    if shutil.which("rocm-smi") is None:
        return out

    # Try JSON (newer ROCm) first
    js = _run(["rocm-smi", "--json"])
    if js:
        try:
            data = json.loads(js)
            # Common structure: {"card": [{"Card Number": "0", "GPU ID": "...", "VRAM Total Memory (B)": ...}, ...]}
            cards = data.get("card") or []
            for c in cards:
                rec = {
                    "vendor": "AMD",
                    "name": c.get("Card SKU") or c.get("Product Name") or "AMD GPU",
                    "pci_bus_id": c.get("PCI Bus") or c.get("Pci BDF") or c.get("PCIe Bus"),
                    "uuid": c.get("GPU ID") or c.get("Unique ID"),
                    "memory_total": None,
                    "memory_free": None,
                }
                # Memory (bytes) → MB
                vt = c.get("VRAM Total Memory (B)") or c.get("vram_total_bytes")
                vf = c.get("VRAM Free Memory (B)") or c.get("vram_free_bytes")
                try:
                    rec["memory_total"] = int(int(vt) // (1024**2)) if vt is not None else None
                except Exception:
                    pass
                try:
                    rec["memory_free"] = int(int(vf) // (1024**2)) if vf is not None else None
                except Exception:
                    pass
                out.append(rec)
            return out
        except Exception:
            pass

    # Fallback to CSV-like flags
    prod = _run(["rocm-smi", "--showproductname"])
    mem = _run(["rocm-smi", "--showmeminfo", "vram"])
    bus = _run(["rocm-smi", "--showbus"])
    if not prod:
        return out

    # Very rough parsing
    products = [ln.split(":")[-1].strip() for ln in prod.splitlines() if "Card series" in ln or "Product" in ln]
    buses = [ln.split(":")[-1].strip() for ln in (bus or "").splitlines() if "PCIe Bus" in ln or "Bus" in ln]
    vram_total = []
    vram_free = []
    if mem:
        for ln in mem.splitlines():
            if "Total Memory" in ln:
                try:
                    mb = int(re.findall(r"(\d+)\s*MB", ln)[0])
                except Exception:
                    mb = None
                vram_total.append(mb)
            if "Used Memory" in ln:
                # free ~ total - used (best-effort)
                try:
                    used = int(re.findall(r"(\d+)\s*MB", ln)[0])
                except Exception:
                    used = None
                if vram_total and used is not None and vram_total[-1] is not None:
                    vram_free.append(max(vram_total[-1] - used, 0))
    n = max(len(products), len(buses), len(vram_total))
    for i in range(n):
        out.append({
            "vendor": "AMD",
            "name": products[i] if i < len(products) else "AMD GPU",
            "pci_bus_id": buses[i] if i < len(buses) else None,
            "uuid": None,
            "memory_total": vram_total[i] if i < len(vram_total) else None,
            "memory_free": vram_free[i] if i < len(vram_free) else None,
        })
    return out

def _torch_info() -> Dict[str, Any]:
    """
    Summarize PyTorch backends (CUDA & Apple Metal MPS).
    """
    info = {
        "torch_available": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cuda_devices": [],
        "apple_mps_available": False,
    }
    try:
        import torch
        info["torch_available"] = True
        info["torch_version"] = torch.__version__
        # CUDA
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if info["cuda_available"]:
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                # memory stats require a context—guard in try
                total_mb = int(prop.total_memory // (1024**2))
                try:
                    with torch.cuda.device(i):
                        res = torch.cuda.mem_get_info()
                        free_mb = int(res[0] // (1024**2))
                except Exception:
                    free_mb = None
                info["cuda_devices"].append({
                    "index": i,
                    "name": prop.name,
                    "multi_processor_count": prop.multi_processor_count,
                    "compute_capability": f"{prop.major}.{prop.minor}",
                    "total_memory_mb": total_mb,
                    "free_memory_mb": free_mb,
                    "pci_bus_id": getattr(prop, "pci_bus_id", None) if hasattr(prop, "pci_bus_id") else None
                })
        # Apple Metal MPS
        try:
            info["apple_mps_available"] = getattr(torch.backends.mps, "is_available", lambda: False)()
        except Exception:
            info["apple_mps_available"] = False
    except Exception:
        pass
    return info

def _likely_nvidia_mps_supported(cuda_devices: List[Dict[str, Any]]) -> bool:
    """
    Heuristic for NVIDIA CUDA MPS (Multi-Process Service) support:
    - Linux only
    - 'nvidia-cuda-mps-control' present
    - Any device has compute capability >= 3.5
    (This is a best-effort check; runtime config still required to enable MPS.)
    """
    if platform.system() != "Linux":
        return False
    if shutil.which("nvidia-cuda-mps-control") is None:
        return False
    # If we have compute capability info, use it
    for d in cuda_devices:
        cc = d.get("compute_capability")
        try:
            if cc:
                major, minor = map(int, cc.split("."))
                if (major, minor) >= (3, 5):
                    return True
        except Exception:
            continue
    # If we don't know CC but tool exists on Linux, say "possibly"
    return False

def get_gpu_inventory() -> Dict[str, Any]:
    """
    Returns:
      {
        "devices": [ { per-GPU fields... } ],
        "backends": {
            "torch_available": bool,
            "torch_version": str|None,
            "cuda_available": bool,
            "cuda_version": str|None,
            "apple_mps_available": bool,
            "nvidia_mps_possible": bool
        }
      }
    """
    torch_info = _torch_info()
    devices = []

    # Prefer PyTorch’s CUDA list for NVIDIA when present (richer CC info)
    if torch_info["cuda_available"] and torch_info["cuda_devices"]:
        for d in torch_info["cuda_devices"]:
            devices.append({
                "vendor": "NVIDIA",
                "name": d["name"],
                "uuid": None,  # fill from nvidia-smi/NVML if needed
                "pci_bus_id": d.get("pci_bus_id"),
                "memory_total": d["total_memory_mb"],
                "memory_free": d["free_memory_mb"],
                "compute_capability": d["compute_capability"],
                "driver_version": None,
                "cuda_version": torch_info["cuda_version"],
            })

    # Merge/augment with NVML and nvidia-smi if available
    def _merge(devs: List[Dict[str, Any]]):
        # try to align by bus_id or name
        for nd in devs:
            matched = None
            for cur in devices:
                if cur.get("pci_bus_id") and nd.get("pci_bus_id") and cur["pci_bus_id"] == nd["pci_bus_id"]:
                    matched = cur; break
                if cur["name"] == nd.get("name"):
                    matched = cur; break
            if matched:
                for k, v in nd.items():
                    if k not in matched or matched[k] in (None, "", 0):
                        matched[k] = v
            else:
                devices.append(nd)

    _merge(_nvml_info())
    _merge(_parse_nvidia_smi())
    # AMD/ROCm devices
    amd = _rocm_info()
    for a in amd:
        devices.append(a)

    backends = {
        "torch_available": torch_info["torch_available"],
        "torch_version": torch_info["torch_version"],
        "cuda_available": torch_info["cuda_available"],
        "cuda_version": torch_info["cuda_version"],
        "apple_mps_available": torch_info["apple_mps_available"],
        "nvidia_mps_possible": _likely_nvidia_mps_supported(torch_info.get("cuda_devices", [])),
    }

    return {"devices": devices, "backends": backends}


# ---------- Persistence helpers ----------
def _settings_path(settings_dir: Path) -> Path:
    settings_dir.mkdir(parents=True, exist_ok=True)
    return settings_dir / "gpu_selection.json"

def load_saved_selection(settings_dir: Path) -> Dict[str, Any]:
    p = _settings_path(settings_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}

def save_selection(settings_dir: Path, data: Dict[str, Any]) -> None:
    p = _settings_path(settings_dir)
    p.write_text(json.dumps(data, indent=2))


# ---------- Device choice + container args ----------
def choose_device_for_user(
    inventory: Dict[str, Any],
    selector: Optional[str] = None
) -> Dict[str, Any]:
    """
    selector can be:
      - None or 'auto'
      - 'idx:<int>'
      - 'uuid:<GPU-...>'
      - 'name:<substring>'
    """
    devs = inventory.get("devices", [])
    if not devs:
        return {}

    def norm(s): return (s or "").lower()

    if not selector or selector == "auto":
        nvidia = [d for d in devs if norm(d.get("vendor")) == "nvidia"]
        if nvidia:
            return nvidia[0]
        amd = [d for d in devs if norm(d.get("vendor")) == "amd"]
        return amd[0] if amd else devs[0]

    if selector.startswith("idx:"):
        want = selector.split(":", 1)[1]
        for d in devs:
            if str(d.get("index")) == want:
                return d

    if selector.startswith("uuid:"):
        want = selector.split(":", 1)[1]
        for d in devs:
            if d.get("uuid") == want:
                return d

    if selector.startswith("name:"):
        want = norm(selector.split(":", 1)[1])
        for d in devs:
            if want in norm(d.get("name")):
                return d

    return devs[0]


def build_container_gpu_selection(
    device: Dict[str, Any],
    runtime: str = "docker",
    vendor_hint: Optional[str] = None
) -> Tuple[List[str], Dict[str, str]]:
    """
    Return (extra_args, extra_env) for container launch.

    runtime: 'docker' | 'singularity' | 'apptainer'
    """
    vendor = (device.get("vendor") or vendor_hint or "").upper()
    idx = device.get("index")
    gpu_uuid = device.get("uuid")

    extra_args: List[str] = []
    extra_env: Dict[str, str] = {}

    if runtime == "docker":
        if vendor == "NVIDIA":
            selector = gpu_uuid or (str(idx) if idx is not None else None)
            if selector is None:
                # fallback: all GPUs
                return (["--gpus", "all"], extra_env)
            # Prefer explicit device selection
            extra_args.extend(["--gpus", f'"device={selector}"'])
            # Alternative env approach:
            # extra_args.extend(["--gpus", "all"])
            # extra_env["NVIDIA_VISIBLE_DEVICES"] = selector

        elif vendor == "AMD":
            # Typical ROCm passes + HIP visibility by index
            extra_args.extend(["--device=/dev/kfd", "--device=/dev/dri", "--group-add=video"])
            if idx is not None:
                extra_env["HIP_VISIBLE_DEVICES"] = str(idx)

    elif runtime in ("singularity", "apptainer"):
        if vendor == "NVIDIA":
            extra_args.append("--nv")
            if idx is not None:
                extra_env["CUDA_VISIBLE_DEVICES"] = str(idx)
            elif gpu_uuid:
                # Some stacks respect UUID, index is safer
                extra_env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        elif vendor == "AMD":
            extra_args.append("--rocm")
            if idx is not None:
                extra_env["HIP_VISIBLE_DEVICES"] = str(idx)

    return extra_args, extra_env


# ---------- Formatting helpers ----------
def _short_uuid(u: Optional[str]) -> str:
    if not u:
        return ""
    return u if len(u) <= 20 else u[:10] + "…" + u[-6:]

def _row_label(d: Dict[str, Any]) -> str:
    parts = []
    if d.get("index") is not None:
        parts.append(f"idx:{d['index']}")
    if d.get("uuid"):
        parts.append(f"uuid:{_short_uuid(d['uuid'])}")
    parts.append(d.get("name") or "GPU")
    return " | ".join(parts)

def _make_selector_value(d: Dict[str, Any]) -> str:
    if d.get("uuid"):
        return f"uuid:{d['uuid']}"
    if d.get("index") is not None:
        return f"idx:{d['index']}"
    return f"name:{(d.get('name') or '').strip()}"


# ---------- Streamlit widget ----------
def render(
    settings_dir: str,
    runtime_default: str = "docker",
    title: str = "GPU Selection"
) -> None:
    """
    Display inventory, allow user to choose a GPU, persist choice,
    and preview container args/env.

    settings_dir: directory path where gpu_selection.json will be stored
    runtime_default: 'docker' | 'singularity' | 'apptainer'
    """
    settings_path = Path(settings_dir)
    saved = load_saved_selection(settings_path)
    st.header(title)

    # Inventory
    inv = get_gpu_inventory()
    devices = inv.get("devices", [])

    if not devices:
        st.warning("No GPUs detected. You can still proceed (CPU-only).")
        # Clear selection if any
        if saved:
            save_selection(settings_path, {})
        return

    # Build a compact table
    table_rows = []
    for d in devices:
        table_rows.append({
            "Label": _row_label(d),
            "Vendor": d.get("vendor"),
            "Name": d.get("name"),
            "Index": d.get("index"),
            "UUID": d.get("uuid"),
            "Bus": d.get("pci_bus_id"),
            "Mem MB (tot)": d.get("memory_total"),
            "Mem MB (free)": d.get("memory_free"),
            "CC": d.get("compute_capability"),
            "Driver": d.get("driver_version"),
            "CUDA": d.get("cuda_version"),
        })
    st.dataframe(table_rows, use_container_width=True)

    # Runtime choice
    runtime = st.selectbox(
        "Container runtime",
        options=["docker", "apptainer", "singularity"],
        index=["docker", "apptainer", "singularity"].index(runtime_default),
        help="Used to preview the correct flags/env for your jobs."
    )

    # Build selection options
    opts = ["auto"]
    display = ["Auto (prefer NVIDIA, else AMD, else first)"]
    for d in devices:
        val = _make_selector_value(d)
        opts.append(val)
        display.append(_row_label(d))

    # Default selector from saved file (if valid), else "auto"
    default_selector = saved.get("selector", "auto")
    if default_selector not in opts:
        # Try to map saved cache fields to a current device
        cached = saved.get("cached_device", {})
        # UUID first
        uu = cached.get("uuid")
        if uu and any(x for x in devices if x.get("uuid") == uu):
            default_selector = f"uuid:{uu}"
        elif cached.get("index") is not None and any(str(x.get("index")) == str(cached.get("index")) for x in devices):
            default_selector = f"idx:{cached['index']}"
        else:
            default_selector = "auto"

    selector = st.selectbox(
        "Choose GPU",
        options=opts,
        format_func=lambda v: display[opts.index(v)],
        index=opts.index(default_selector)
    )

    # Resolve device now
    chosen = choose_device_for_user(inv, selector)
    if not chosen:
        st.error("Could not resolve a device from selection.")
        return

    # Persist on change
    selection_payload = {
        "selector": selector,
        "cached_device": {
            "vendor": chosen.get("vendor"),
            "name": chosen.get("name"),
            "uuid": chosen.get("uuid"),
            "index": chosen.get("index"),
            "pci_bus_id": chosen.get("pci_bus_id"),
        },
        "runtime": runtime,
    }

    # Save button (explicit to avoid overwriting on every rerun)
    if st.button("Save GPU Preference"):
        save_selection(settings_path, selection_payload)
        st.success(f"Saved to { _settings_path(settings_path) }")

    # Preview container args/env
    extra_args, extra_env = build_container_gpu_selection(chosen, runtime=runtime)

    st.subheader("Container launch preview")
    st.code(f"Runtime: {runtime}\nExtra args: {extra_args}\nExtra env:  {extra_env}", language="bash")

    if runtime == "docker":
        # Quick preview command (illustrative)
        cmd = ["docker", "run"] + extra_args
        for k, v in extra_env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += ["<image>", "python", "-c", "'print(123)'"]
        st.caption("Example:")
        st.code(" ".join(cmd), language="bash")
    else:
        # Apptainer/Singularity preview
        env_parts = []
        for k, v in extra_env.items():
            env_parts += ["--env", f"{k}={v}"]
        cmd = ["apptainer" if runtime == "apptainer" else "singularity", "exec"] + env_parts + extra_args + ["image.sif", "python", "-c", "'print(123)'"]
        st.caption("Example:")
        st.code(" ".join(cmd), language="bash")


# ---------- Convenience API for the rest of your app ----------
def load_container_selection(
    settings_dir: str,
    runtime_override: Optional[str] = None
) -> Tuple[List[str], Dict[str, str], Dict[str, Any]]:
    """
    Load persisted selection and return:
      (extra_args, extra_env, chosen_device)
    If nothing saved, falls back to 'auto' and does NOT persist.
    """
    settings_path = Path(settings_dir)
    saved = load_saved_selection(settings_path)
    inv = get_gpu_inventory()
    selector = saved.get("selector", "auto")
    runtime = runtime_override or saved.get("runtime", "docker")
    chosen = choose_device_for_user(inv, selector)
    extra_args, extra_env = build_container_gpu_selection(chosen, runtime=runtime)
    return extra_args, extra_env, chosen


def panel_select_gpu():
    render(st.session_state['out_dir'])
