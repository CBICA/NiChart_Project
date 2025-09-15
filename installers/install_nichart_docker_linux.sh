#!/usr/bin/env bash
## This is the Linux NiChart Installation script which installs NiChart Docker containers on your computer. You need Docker installed and running for this to work.
set -Eeuo pipefail

###############################################################################
# Config
###############################################################################
# Hardcoded image for both installer and runtime
APP_IMAGE="cbica/nichart:09122025"

# Command inside the installer container
INSTALLER_CMD="python /app/resources/pull_containers.py /app/resources/tools/"

# Docker run args (installer)
INSTALLER_RUN_ARGS=(--privileged --user 1000 -it --rm --name nichart_installer --entrypoint="/usr/bin/_entrypoint.sh" -v /usr/bin/docker:/usr/bin/docker -v /var/run/docker.sock:/var/run/docker.sock)

###############################################################################

msg()  { echo -e "[\e[1;34mINFO\e[0m] $*"; }
err()  { echo -e "[\e[1;31mERROR\e[0m] $*" >&2; }
die()  { err "$*"; exit 1; }

trap 'err "Installation failed on line $LINENO while running: $BASH_COMMAND"; exit 1' ERR

usage() {
  cat <<'USAGE'
Usage:
  ./install_nichart_docker_linux.sh <DATA_DIR>

Arguments:
  DATA_DIR   Path where the app should store data.
             This path will be embedded in the generated run_nichart.sh.
USAGE
}

# --- Arg parsing ---
if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

DATA_DIR_RAW="$1"
DATA_DIR="$(mkdir -p -- "$DATA_DIR_RAW" && cd "$DATA_DIR_RAW" && pwd -P)" || die "Invalid data directory."
[[ -n "${DATA_DIR:-}" ]] || die "Could not resolve data directory path."

# --- Pre-flight checks ---
command -v docker >/dev/null 2>&1 || die "Docker is not installed or not on PATH."
docker info >/dev/null 2>&1 || die "Docker daemon not reachable. Ensure Docker is running."

# --- Pull app image ---
msg "Pulling image: ${APP_IMAGE}"
docker pull "${APP_IMAGE}" >/dev/null || die "Failed to pull ${APP_IMAGE}"

# --- Run installer container ---
msg "Running installer container..."
msg "Docker run args: ${INSTALLER_RUN_ARGS[@]} ${APP_IMAGE} ${INSTALLER_CMD}"
docker run ${INSTALLER_RUN_ARGS[@]} ${APP_IMAGE} ${INSTALLER_CMD} || die "Installer container failed."

# --- Generate run.sh ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
RUN_SH_PATH="${SCRIPT_DIR}/run_nichart.sh"

cat > "${RUN_SH_PATH}" <<'RUNSH'
#!/usr/bin/env bash
set -Eeuo pipefail

msg()  { echo -e "[\e[1;34mINFO\e[0m] $*"; }
err()  { echo -e "[\e[1;31mERROR\e[0m] $*" >&2; }
die()  { err "$*"; exit 1; }

trap 'err "Run failed on line $LINENO while running: $BASH_COMMAND"; exit 1' ERR

# --- Embedded by install.sh ---
APP_IMAGE="__APP_IMAGE__"
DATA_DIR="__DATA_DIR__"
APP_URL_DEFAULT="http://localhost:8501/"

CONTAINER_NAME="nichart_server"

RUN_ARGS=(--rm -d --name "${CONTAINER_NAME}")
RUN_ARGS+=(--privileged --user 1000 -p 8501:8501 -v /usr/bin/docker:/usr/bin/docker -v /var/run/docker.sock:/var/run/docker.sock)
RUN_ARGS+=(-v "${DATA_DIR}:/app/output_folder:rw")

APP_CMD=()

is_wsl() {
  # WSL 1/2 typically expose "Microsoft" in kernel release; also /proc/version works
  if grep -qi "microsoft" /proc/sys/kernel/osrelease 2>/dev/null; then
    return 0
  elif grep -qi "microsoft" /proc/version 2>/dev/null; then
    return 0
  else
    return 1
  fi
}

open_browser() {
  local url="$1"
  if is_wsl; then
    # Use Windows PowerShell to open the default browser from WSL
    if command -v powershell.exe >/dev/null 2>&1; then
      powershell.exe -NoProfile -Command "Start-Process \"$url\"" >/dev/null 2>&1 || warn "Failed to open browser via PowerShell."
    else
      warn "powershell.exe not found. Please open: $url"
    fi
  else
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$url" >/dev/null 2>&1 || warn "Failed to open browser via xdg-open."
    else
      warn "xdg-open not found. Please open: $url"
    fi
  fi
}

msg "Running command: docker run ${RUN_ARGS[@]} ${APP_IMAGE} ${APP_CMD[@]} $@"
# Start container
msg "Starting container '${CONTAINER_NAME}' from image '${APP_IMAGE}'..."
CID="$(docker run "${RUN_ARGS[@]}" "${APP_IMAGE}" "${APP_CMD[@]}" "$@" || die "docker run failed.")"

# Optionally wait a short time for service to be ready (tune or remove)
sleep 2

# Try to open the browser
msg "Opening: ${APP_URL}"
open_browser "${APP_URL}"

# Show status and friendly tips
if [[ -n "${CID}" ]]; then
  msg "Container ID: ${CID}"
  msg "Use 'docker logs -f ${CONTAINER_NAME}' to tail logs."
  msg "Use 'docker stop ${CONTAINER_NAME}' to force-stop the app."
fi
RUNSH

# Replace placeholders
sed -i "s|__APP_IMAGE__|${APP_IMAGE}|g" "${RUN_SH_PATH}"
sed -i "s|__DATA_DIR__|${DATA_DIR}|g"   "${RUN_SH_PATH}"

chmod +x "${RUN_SH_PATH}"

msg "Installation complete."
msg "Generated launcher: ${RUN_SH_PATH}"
msg "Data directory:     ${DATA_DIR}"
msg "Try: ${RUN_SH_PATH}"
