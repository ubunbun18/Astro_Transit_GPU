# AstroTransit-GPU: Candidate Vetting Pipeline (V39)

The AstroTransit-GPU Vetting Pipeline is a 3-stage intelligent workflow designed to rapidly triage massive amounts of candidates from high-speed GPU screening into high-confidence planet candidates.

## 1. Workflow Overview

The pipeline consists of three main commands:

### Step 1: Sector Screening
Performs a high-speed scan across the entire sector to extract the primary candidate for each target.
```bash
python -m astrotransit_gpu.cli screen-sector --cache-dir data/s1_cache --out s1_raw.csv
```

### Step 2: Candidate Refinement
Performs a deep, multi-candidate search for high-potential targets based on sophisticated selection rules (SNR, Catalogs, Heuristics).
```bash
python -m astrotransit_gpu.cli refine --results s1_raw.csv --cache-dir data/s1_cache --out s1_refined.csv --config configs/vetting_v1.yaml
```

### Step 3: Vetting & Dashboard Generation
Handles harmonic grouping, automated scoring, plot generation, and dashboard assembly.
```bash
python -m astrotransit_gpu.cli vet --results s1_refined.csv --cache-dir data/s1_cache --out reports/s1_vet --config configs/vetting_v1.yaml
```

## 2. Output Artifacts

The output directory contains the following files, ensuring full scientific reproducibility:

- `index.html`: **Triage Dashboard** (Recommended: Open in browser to sort and triage candidates).
- `summary.json`: Machine-readable execution statistics and metadata.
- `candidates_ranked.csv`: The complete ranked list of candidates.
- `plots/`: High-resolution folded light curves (with Raw + Binned views).

## 3. Triage Dashboard Features

The dashboard (`index.html`) is designed to "reduce what the human needs to see" by highlighting priorities.

- **Summary Cards**: Real-time stats for Unknowns, High-score candidates, and Known TOIs.
- **Status Badges**: Color-coded types: `TOI` (Green), `EB` (Red), `Unknown` (Blue).
- **Harmonic Flag (H)**: Automatically marks candidates that are likely harmonic duplicates.
- **Automated Notes**: Displays rationale for scores (e.g., "High Priority", "Short Transit").
- **Robust Preview**: Modal-based plot viewer with fallback handling for candidates outside the Top-N limit. Works offline.

## 4. Selection & Scoring Rules (`vetting_v1.yaml`)

Refinement utilizes 6 intelligent selection rules to rescue/extract targets:
1. **SNR Threshold**: Standard absolute signal strength.
2. **Top-N**: Relative ranking to capture the best signals in the sector.
3. **Catalog Match**: Ensures known TOIs/EBs are included for verification.
4. **Artifact/EB Heuristics**: Captures deep transits or long durations.
5. **Planet-like Heuristics**: Rescues small/shallow transits even at low SNR.
6. **Random Sampling**: Prevents statistical bias in the discovery set.

---

## Scientific Reproducibility
The generated `summary.json` records the Kernel version (V39 Apex Predator), Config path, and Input source.
```json
{
    "kernel_version": "V39 Apex Predator",
    "config": "configs/vetting_v1.yaml",
    "total_targets": 3560,
    "high_score_unknowns": 1956
}
```
