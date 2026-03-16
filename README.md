# dreampy

Pseudobulk mixed-model differential expression for single-cell RNA-seq,
in Python. A port of the R [dreamlet](https://github.com/GabrielHoffman/dreamlet)
framework into the scanpy/AnnData ecosystem.

## Why dreampy?

If your analysis lives in Python (scanpy, AnnData, scverse) but your DE
step requires switching to R, dreampy eliminates that context switch.
It implements the full dreamlet/voom/limma pipeline — pseudobulk
aggregation, TMM normalization, voom precision weights, linear mixed
models with Satterthwaite degrees of freedom, and empirical Bayes
moderation — validated to near-identical output against R dreamlet v1.8.0.

## Installation

Requires Python ≥ 3.12. We recommend using conda to manage the
environment, as one dependency (`scikit-misc`) has broken version
metadata when installed via pip.

**Recommended: conda environment**
```bash
git clone https://github.com/sbwells22/dreampy.git
cd dreampy
conda env create -f envs/env.yml
conda activate dreampy
pip install -e .
```

**Alternatively**, if you prefer to manage your own environment:
```bash
git clone https://github.com/sbwells22/dreampy.git
cd dreampy
conda install -c conda-forge scikit-misc
pip install -e .
```

## Quick start
```python
import scanpy as sc
import dreampy as dp

# Load cell-level data
adata = sc.read_h5ad("my_data.h5ad")

# Pseudobulk aggregation
pb = dp.aggregate_pseudobulk(adata, groupby=["celltype", "donor_id", "site"])

# Preprocessing
pb = dp.filter_samples(pb, min_cells=5, min_samples=3)
pb = dp.compute_tmm_factors(pb, assay_col="assays")
assays = dp.filter_by_expr(pb, assay_col="assays")

# Per-cell-type DE analysis
formula = "~ condition + (1|site)"
for name, assay_pb in assays.items():
    assay_pb = dp.log2cpm(assay_pb)
    assay_pb = dp.estimate_weights(assay_pb, formula=formula)
    fit = dp.fit_models(assay_pb, formula=formula)
    fit_eb = dp.ebayes(fit)
    results = dp.get_results(fit_eb, assay_name=name)
```

See [`notebooks/tutorial.py`](notebooks/tutorial.py) for a complete
walkthrough with synthetic data included in the repository.

## Validation

dreampy is validated against R dreamlet v1.8.0 on two published datasets
(Wells 2025, Perez 2022) with Pearson r ≥ 0.999 for all pipeline
quantities in assays with n ≥ 100 samples. See the [preprint]() for
details.

<details>
<summary>R validation environment setup (developers only)</summary>

The R validation arm is only needed to regenerate reference outputs.

1. Create the conda environment (includes R and Python):
```bash
   # Uncomment the R section!
   conda env create -f envs/env.yml
   conda activate dreampy
```

2. Install pinned R packages:
```bash
   Rscript r-reference/install_packages.R
```

3. Apple Silicon users may need:
```bash
   mkdir -p ~/.R && echo -e "CC17 = clang\nCC17STD = -std=gnu17" > ~/.R/Makevars
```

</details>

## Citation

If you use dreampy in your research, please cite:

> Wells, S.B. et al. (2026). dreampy: ... *bioRxiv*. doi: ...

## License

MIT