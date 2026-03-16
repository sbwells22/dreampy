# R Reference Generation Scripts

These scripts generate the frozen CSV reference artifacts used by dreampy's
test suite. **CI never runs these scripts.** They are provenance documentation
showing exactly how reference values were produced.

## Environment

Fill in after running. All fields are required for reproducibility.

```
R version:           4.5.2 (2025-10-31)
Bioconductor:        3.22
Platform:            aarch64-apple-darwin20.0.0

Packages:
  dreamlet             1.8.0
  variancePartition    1.40.0
  limma                3.66.0
  edgeR                4.8.0
  lme4                 1.1.38
  lmerTest             3.1.3
  SingleCellExperiment 1.32.0
```

Scripts 3 and 4 print their session info to stdout — copy it here.

## Seed

All scripts use `set.seed(11)` — dreampy signature convention.

## Usage

Run from this directory (`tests/R/`):

```bash
# 1. Generate synthetic count matrices and metadata (no Bioconductor needed)
Rscript generate_synthetic_data.R

# 2. Generate Design A (FE) references (needs: edgeR, limma)
Rscript generate_design_a_refs.R

# 3. Generate Design B (single RE) references (needs: dreamlet + deps)
Rscript generate_design_b_refs.R

# 4. Generate Design C (multi RE) references (needs: dreamlet + deps)
Rscript generate_design_c_refs.R
```

Scripts must be run in order. Each writes to `../data/synthetic/design_{a,b,c}/`.

An optional `OUTDIR`/`DATADIR` argument overrides the default path:

```bash
Rscript generate_synthetic_data.R /path/to/output
Rscript generate_design_a_refs.R  /path/to/output/design_a
```

## Script dependency graph

```
generate_synthetic_data.R          ← no external deps beyond base R
    │
    ├── generates design_a/{counts,metadata,gene_manifest}.csv
    ├── generates design_b/{counts,metadata,gene_manifest}.csv
    └── generates design_c/{counts,metadata,gene_manifest}.csv
            │               │               │
            ▼               ▼               ▼
  generate_design_a_refs.R  │               │    ← edgeR, limma
                generate_design_b_refs.R    │    ← dreamlet, variancePartition, ...
                            generate_design_c_refs.R  ← dreamlet, variancePartition, ...
```

## What each script exports

### Script 1: `generate_synthetic_data.R`

For each design: `counts.csv`, `metadata.csv`, `gene_manifest.csv`.

No statistical analysis — pure data generation.

### Script 2: `generate_design_a_refs.R`

| File | Contents |
|------|----------|
| `ref_filter_mask.csv` | Per-gene boolean: retained by `filterByExpr`? |
| `ref_tmm_factors.csv` | Per-sample: `lib_size`, `norm_factors` |
| `ref_log2cpm.csv` | Genes × samples log2 CPM matrix (post-filter) |
| `ref_voom_weights.csv` | Genes × samples precision weights (post-filter) |
| `ref_lmfit.csv` | Per-gene: `sigma`, `coef_*`, `stdev_unscaled_*` |
| `ref_ebayes.csv` | Per-gene: `s2_prior`, `df_prior`, `s2_post`, `df_total`, `t_*`, `p_*`, `B_*`, `adj_p_*` |

### Scripts 3 & 4: `generate_design_{b,c}_refs.R`

Same files as Design A, plus:

| File | Contents |
|------|----------|
| `ref_lmer.csv` | Per-gene: `sigma`, `rdf` (scalar residual df for eBayes), `edf` (effective params), `coef_*`, `stdev_unscaled_*`, `varComp_*` (variance components) |
| `ref_satterthwaite.csv` | Per-gene: `rdf` (scalar), `df_residual_*` (per-coefficient Satterthwaite df, pre-eBayes) |
| `ref_lmer_kr.csv` | **Design B only.** Same schema as `ref_lmer.csv` but with KR bias-corrected `stdev_unscaled_*`. Coefficients and sigma should match the Satterthwaite version. |
| `ref_ebayes_kr.csv` | **Design B only.** Same schema as `ref_ebayes.csv` but with KR-derived df, t, p, b, adj_p. |
| `ref_kr_df.csv` | **Design B only.** Per-gene: `rdf`, `df_residual_*` (per-coefficient KR denominator df, pre-eBayes). |

Note: `ref_lmfit.csv` is not produced for Designs B/C. The mixed model results go in `ref_lmer.csv`.

In the eBayes output for Designs B/C, `df_total` is exported per-coefficient (`df_total_*`) because
Satterthwaite df varies across coefficients within each gene. This differs from Design A where
`df_total` is a single per-gene value.

## Scaffold+swap approach (Designs B and C)

dreamlet's `processAssays` expects a `SingleCellExperiment` with specific internal
metadata structures (`int_colData$n_cells`, `metadata$agg_pars`, `metadata$aggr_means`).
Rather than construct these manually, we:

1. Create a fake cell-level SCE with trivial `rpois(lambda=10)` counts
2. Call `aggregateToPseudoBulk()` to build the correct scaffold
3. Swap `assay(pb, "celltype_A")` with the designed counts from `counts.csv`

This guarantees all internal structures match what dreamlet expects. The fake
cell counts only affect the `min.cells` filter — we pass `min.cells=1` to
`processAssays` to prevent spurious sample exclusion.

## Troubleshooting

**"Package X not found"**: Install Bioconductor packages with:
```r
BiocManager::install(c("dreamlet", "variancePartition", "edgeR", "limma",
                        "SingleCellExperiment"))
```

**Convergence warnings on Design B/C edge-case genes**: Expected. Genes 226-230
(Design B) and 266-270 (Design C) have zero true RE variance. lmer may emit
boundary warnings. The scripts capture results regardless — the convergence
status is recorded in `ref_lmer.csv`.

**Different results after R/package upgrade**: This is why the CSVs are
committed artifacts. If you re-run and get different numbers, the existing
committed CSVs remain the ground truth for CI. Update only after careful
validation that the new R results are correct.

**varComp extraction warnings**: The accessor for variance components varies
across dreamlet/variancePartition versions. The scripts try `fit$varComp`
first, then `fit$theta` as fallback. If neither works, you may need to
inspect the object structure with `str(fit)` and adapt the extraction code.
