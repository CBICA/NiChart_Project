# GBM DNA-methylation cell-state deconvolution

## Summary

Takes a directory of Illumina methylation-array IDAT files, preprocesses them

with `minfi` (auto-detecting 450k/EPIC/EPICv2), and runs reference-based

cell-state deconvolution (`EpiDISH`, RPC) against bundled GBM cell-state

profiles. Outputs per-sample malignant/non-malignant fractions, a tumor-purity

estimate, and an MCcsi stem-vs-differentiated class per sample.


## Input

- One directory of Illumina IDAT pairs, named `{MRID}_Grn.idat` + `{MRID}_Red.idat`

(the standard Illumina naming; MRID = the shared sample/scan id).

- Array platform is **auto-detected per sample** by IDAT file size

(450k / EPIC / EPICv2) — the caller does not need to specify array type.

- A run may contain a mix of platforms; each is processed with the appropriate

minfi pipeline and the results are merged on shared CpGs.

## Output (written into `--output`)

- `predicted.cell.states.csv` — the only output, one row per sample:

- cell-state fractions: `stem_like, diff2, diff1, immune, neuron, glia`

- normalized malignant fractions: `stem_like_normalized, diff2_normalized, diff1_normalized`

- `tumor_purity` (sum of malignant fractions)

- `stem_like_normalized_rounded` and `MCcsi_feat` (class: `DIF` / `unclassified` / `STM`)

Intermediate beta-value matrices and QC are computed in memory but not written out.

## Resources (approximate)

- CPU: scales with `--cores`; preprocessing is the heavy step.

- RAM: roughly 2-4 GB per worker core; EPICv2 (`preprocessQuantile`) is the most

memory-intensive. For a typical batch, ~16 GB total is comfortable; raise/lower

`--cores`/`--batch` to fit the host.

- Runtime: dominated by IDAT reading + normalization, ~seconds to minutes per sample.

## Notes

No patient data is in the image–only public reference resources

(Zhou-lab masks; GBM cell-state profiles, published at

https://github.com/danasilv/Deconvolution_of_GBM_bulk_DNA_methylation_profiles)