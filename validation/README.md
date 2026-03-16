# Validation

Real-data validation of dreampy against R dreamlet. Reproduces the full pipeline — pseudobulk aggregation through eBayes — on two published datasets and compares every intermediate quantity against R reference outputs.

For the synthetic test suite (CI, no external data), see `tests/`.

## Datasets

**Wells et al. 2025** — T cell subset from the Immune Aging study. 13 T cell types, ~135 pseudobulk samples (donor × tissue), formula `~ age_groups + (1|donor_id) + (1|tissue_abbrev)`. Primary cross-language validation dataset.

**Perez et al. 2022** — SLE lupus (GSE174188). 10 immune cell types, ~336 pseudobulk samples, 261 donors, formula `~ sle + (1|donor_id) + (1|Processing_Cohort)`. Scaling validation and biological reanalysis. Includes ImmVar sensitivity analysis.

## Directory Structure

```
validation/
├── data/
│   ├── perez2022/          # Generated CSVs and downloaded h5ad (gitignored)
│   └── wells2025/          # Generated CSVs and downloaded h5ad (gitignored)
├── python/
│   ├── preprocess_perez2022.py
│   ├── preprocess_wells2025.py
│   ├── validate_perez2022.py
│   ├── validate_wells2025.py
│   └── sensitivity_perez2022.py
├── R/
│   ├── reference_perez2022.R
│   └── reference_wells2025.R
├── sh/
│   ├── download_perez2022.sh
│   └── download_wells2025.sh
└── README.md
```

All files under `data/` (h5ad downloads and generated CSVs) are gitignored. The scripts regenerate everything from the public data sources.

## Prerequisites

**Python:** dreampy installed with dependencies, plus scanpy for h5ad loading.

**R:** dreamlet, variancePartition, SingleCellExperiment, edgeR, limma, lme4, zellkonverter, BiocParallel.

## Workflow

Run all commands from the repository root.

### Step 1: Download data

```bash
bash validation/sh/download_wells2025.sh
bash validation/sh/download_perez2022.sh
```

Downloads h5ad files from CellxGene into `validation/data/{dataset}/`. Skips if file already exists. Verifies SHA256 checksum after download.

### Step 2: Preprocess

```bash
python validation/python/preprocess_wells2025.py
python validation/python/preprocess_perez2022.py
```

Loads raw counts from the `.raw` slot, applies gene filtering (11 categories of technical artifacts + protein-coding whitelist), encodes derived covariates, and writes a filtered h5ad consumed by both R and Python pipelines. This ensures both languages start from identical gene-filtered input.

Preprocessing reports are printed to stdout. To save:

```bash
python validation/python/preprocess_wells2025.py > validation/python/wells2025_preprocess_report.txt
python validation/python/preprocess_perez2022.py > validation/python/perez2022_preprocess_report.txt
```

### Step 3: Generate R references

```bash
Rscript validation/R/reference_wells2025.R
Rscript validation/R/reference_perez2022.R
```

Loads the filtered h5ad, pseudobulks via `aggregateToPseudoBulk`, runs `processAssays` + `dreamlet` with Satterthwaite df, and exports per-assay CSVs with all intermediate quantities (TMM factors, filter masks, log2CPM, voom weights, coefficients, variance components, Satterthwaite df, eBayes outputs). Uses all available cores.

### Step 4: Run Python validation

```bash
pytest validation/python/validate_wells2025.py -v
pytest validation/python/validate_perez2022.py -v
```

Runs the full dreampy pipeline on the same filtered input and compares against the R reference CSVs at every stage. Produces a validation metrics CSV (`{dataset}_validation_metrics.csv`) and a timing file.

Expected results (v0.1):
- Wells: 332/351 passing (19 failures in small-sample assays — optimizer boundary sensitivity)
- Perez: 249/270 passing (21 failures — PB collinearity + small-sample assays)

All failures are characterized and documented in the phase 3 plan. No algorithmic errors.

### Step 5 (Perez only): ImmVar sensitivity analysis

```bash
python validation/python/sensitivity_perez2022.py
```

Re-runs the full dreampy pipeline excluding the 50 ImmVar control donors, then computes per-celltype concordance metrics (logFC correlation, t-stat correlation, Jaccard overlap of FDR < 0.05 gene sets) against the full-cohort results from Step 4. Produces `sensitivity_concordance_report.csv`.

## Validation Strategy

Quantities are validated in two tiers:

**Tier 1 (exact match):** TMM normalization factors and log2CPM values involve no optimization or smoothing — they should match R to floating-point precision. Validated with `assert_allclose`.

**Tier 2 (agreement):** Voom weights through eBayes outputs involve lowess smoothing and mixed-model optimization where cross-language differences are expected. Validated with a 7-metric report (mean ratio, median ratio, ratio range, mean % difference, max |% difference|, RMSE, Pearson correlation) plus pass/fail gates (correlation floor + fraction-based tolerance).

## Known Discrepancies
The most significant for validation:

1. **filterByExpr floating-point tie-break** — ~1–2 genes per assay at exact CPM boundary. Validation uses shared gene intersection.
2. **Optimizer divergence** — Cold-start (dreampy) vs warm-start (R dreamlet) on multimodal likelihood surfaces. Both solutions statistically valid.
3. **Design matrix collinearity** — dreampy drops perfectly collinear RE terms; R keeps them and lets the optimizer handle near-singularity. Observed in Perez PB assay. 100% DE concordance on shared genes despite structural model difference.