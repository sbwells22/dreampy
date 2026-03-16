#!/usr/bin/env python3
"""
generate_tutorial_data.py
-------------------------
Generates a synthetic cell-level AnnData (.h5ad) for the dreampy tutorial
notebook. The dataset models a case/control study with a collection-site
random effect and four cell types exhibiting distinct DE profiles.

Experimental design:
    Formula: ~ condition + (1|site)
    40 donors (20 case, 20 control), 4 collection sites, 500 genes.
    ~50 cells per donor per cell type (Poisson-distributed, floored at 10).

Signal layout:
    celltype_A — 30% of genes DE, strong effects (mean |logFC| ≈ 1.0)
    celltype_B — 15% of genes DE, weaker effects (mean |logFC| ≈ 0.6)
    celltype_C — no DE genes (null cell type)
    celltype_D — 30% of genes DE, ~half shared with celltype_A

Cell-type identity is encoded via per-celltype expression scaling: each
cell type has ~10% of genes independently down-scaled to near-zero,
producing cell-type-specific expression profiles and per-assay filtering
differences.

Counts are drawn from a Negative Binomial distribution per cell,
incorporating per-cell library size variation, condition fixed effects,
and per-gene site random effects.

Outputs:
    tutorial_dataset.h5ad — cell-level AnnData (cells × genes, sparse)
    gene_manifest.csv     — ground-truth DE status and logFC per cell type

Usage:
    python generate_tutorial_data.py [OUTDIR]

    OUTDIR defaults to ../data/tutorial (relative to this script)

Seed: 11
"""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

# ── Configuration ─────────────────────────────────────────────────────

SEED = 11

N_DONORS = 40  # 20 case, 20 control
N_SITES = 4
N_GENES = 500
CELLTYPE_NAMES = ["celltype_A", "celltype_B", "celltype_C", "celltype_D"]

# Mean cells per donor per cell type (actual count drawn from Poisson)
MEAN_CELLS_PER_DONOR = 50
MIN_CELLS_PER_DONOR = 10  # floor to avoid degenerate pseudobulk

# DE signal per cell type
CELLTYPE_DE_CONFIG = {
    "celltype_A": {"de_frac": 0.30, "mean_abs_log2fc": 1.0, "sd_log2fc": 0.3},
    "celltype_B": {"de_frac": 0.15, "mean_abs_log2fc": 0.6, "sd_log2fc": 0.2},
    "celltype_C": {"de_frac": 0.00, "mean_abs_log2fc": 0.0, "sd_log2fc": 0.0},
    "celltype_D": {"de_frac": 0.30, "mean_abs_log2fc": 1.0, "sd_log2fc": 0.3},
}

# Fraction of celltype_D's DE genes shared with celltype_A
D_A_OVERLAP_FRAC = 0.5

# Fraction of genes with low expression per cell type (creates
# cell-type-specific expression profiles and filtering differences)
LOW_EXPR_FRAC = 0.10
LOW_EXPR_SCALE_RANGE = (0.01, 0.10)

# Site random-effect SD (log2-scale, applied per gene per cell type)
SITE_RE_SIGMA = 0.5

# Per-cell library size log-normal SD
LIB_SIZE_SIGMA = 0.3

# NB dispersion: size parameter range (dispersion = 1/size)
NB_SIZE_RANGE = (0.5, 5.0)

# Gene base mean range (cell-level; pseudobulk sums ≈ mean × n_cells)
BASE_MEAN_RANGE = (0.05, 3.0)


# ── Helpers ───────────────────────────────────────────────────────────


def assign_de_genes(rng, celltype_configs, n_genes, overlap_frac):
    """
    Assign DE gene indices and log2FC vectors per cell type.

    celltype_D shares `overlap_frac` of its DE genes with celltype_A;
    the rest are drawn from the non-A-DE pool.

    Returns
    -------
    de_indices : dict[str, np.ndarray]
        Per-cell-type arrays of gene indices flagged as DE.
    log2fc : dict[str, np.ndarray]
        Per-cell-type log2FC vectors (length n_genes, 0 for non-DE).
    """
    de_indices = {}
    log2fc = {}

    for ct_name in celltype_configs:
        config = celltype_configs[ct_name]
        n_de = int(n_genes * config["de_frac"])

        if n_de == 0:
            de_indices[ct_name] = np.array([], dtype=int)
            log2fc[ct_name] = np.zeros(n_genes)
            continue

        if ct_name == "celltype_D":
            # Share a fraction with celltype_A
            a_de = de_indices["celltype_A"]
            n_shared = min(int(n_de * overlap_frac), len(a_de))
            shared = rng.choice(a_de, size=n_shared, replace=False)
            remaining_pool = np.setdiff1d(np.arange(n_genes), a_de)
            n_unique = n_de - n_shared
            unique = rng.choice(remaining_pool, size=n_unique, replace=False)
            idx = np.concatenate([shared, unique])
        else:
            idx = rng.choice(n_genes, size=n_de, replace=False)

        de_indices[ct_name] = idx

        # Log2FC: random sign, magnitude drawn from |N(mean, sd)|
        fc = np.zeros(n_genes)
        signs = rng.choice([-1, 1], size=len(idx))
        magnitudes = np.abs(
            rng.normal(config["mean_abs_log2fc"], config["sd_log2fc"], size=len(idx))
        )
        fc[idx] = signs * magnitudes
        log2fc[ct_name] = fc

    return de_indices, log2fc


def make_celltype_scales(rng, celltype_names, n_genes, low_frac, low_range):
    """
    Per-cell-type expression scaling factors (length n_genes each).

    Each cell type independently down-scales ~low_frac of genes to
    near-zero, creating cell-type-specific expression profiles.
    """
    scales = {}
    for ct in celltype_names:
        s = np.ones(n_genes)
        low_idx = rng.choice(n_genes, size=int(n_genes * low_frac), replace=False)
        s[low_idx] = rng.uniform(*low_range, size=len(low_idx))
        scales[ct] = s
    return scales


def make_site_effects(rng, celltype_names, n_genes, site_names, sigma):
    """
    Per-gene, per-site random effects on the log2 scale.

    Returns dict[celltype -> DataFrame(n_genes × n_sites)].
    Fresh draws per cell type so site impact varies across cell types.
    """
    effects = {}
    for ct in celltype_names:
        mat = rng.normal(0, sigma, size=(n_genes, len(site_names)))
        effects[ct] = pd.DataFrame(mat, columns=site_names)
    return effects


def generate_chunk_counts(
    rng, n_cells, base_means, nb_sizes, ct_scale, log2fc,
    site_shift, condition, lib_sigma,
):
    """
    Generate an (n_cells × n_genes) NB count matrix for one
    donor × cell-type combination.

    Parameters
    ----------
    rng : np.random.Generator
    n_cells : int
    base_means : np.ndarray, shape (n_genes,)
    nb_sizes : np.ndarray, shape (n_genes,)
    ct_scale : np.ndarray, shape (n_genes,)
        Cell-type expression scaling.
    log2fc : np.ndarray, shape (n_genes,)
        Condition log2FC (applied only if condition == "case").
    site_shift : np.ndarray, shape (n_genes,)
        Site RE on log2 scale for this donor's site.
    condition : str
        "case" or "control".
    lib_sigma : float
        SD for per-cell log-normal library size factor.

    Returns
    -------
    counts : np.ndarray, shape (n_cells, n_genes), dtype int
    """
    n_genes = len(base_means)

    # Per-cell library size factors
    lib_factors = np.exp(rng.normal(0, lib_sigma, size=n_cells))

    # Gene-level mean: base × cell-type scale × site RE
    mu_gene = base_means * ct_scale * (2.0 ** site_shift)

    # Condition fixed effect (case only)
    if condition == "case":
        mu_gene = mu_gene * (2.0 ** log2fc)

    # Expand to (n_cells, n_genes): outer product of lib factors and gene means
    mu = np.outer(lib_factors, mu_gene)
    mu = np.maximum(mu, 1e-6)

    # NB draw: parameterize as (n, p) where mean = n(1-p)/p → p = n/(n+mu)
    sizes_broadcast = np.broadcast_to(nb_sizes, mu.shape)
    p = sizes_broadcast / (sizes_broadcast + mu)

    counts = rng.negative_binomial(n=sizes_broadcast, p=p)
    return counts


# ── Main ──────────────────────────────────────────────────────────────


def main():
    args = sys.argv[1:]
    outdir = Path(args[0]) if args else Path(__file__).parent / ".." / "data" / "tutorial"
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # ── Donor metadata ────────────────────────────────────────────────

    donor_ids = [f"donor_{i + 1:03d}" for i in range(N_DONORS)]
    conditions = ["case"] * (N_DONORS // 2) + ["control"] * (N_DONORS // 2)

    # Round-robin site assignment for balance
    site_names = [f"site_{chr(65 + i)}" for i in range(N_SITES)]
    donor_sites = [site_names[i % N_SITES] for i in range(N_DONORS)]

    donor_meta = pd.DataFrame({
        "donor_id": donor_ids,
        "condition": conditions,
        "site": donor_sites,
    })

    # ── Gene parameters ───────────────────────────────────────────────

    gene_names = [f"gene_{i + 1:04d}" for i in range(N_GENES)]
    base_means = rng.uniform(*BASE_MEAN_RANGE, size=N_GENES)
    nb_sizes = rng.uniform(*NB_SIZE_RANGE, size=N_GENES)

    # ── Cell-type-specific parameters ─────────────────────────────────

    ct_scales = make_celltype_scales(
        rng, CELLTYPE_NAMES, N_GENES, LOW_EXPR_FRAC, LOW_EXPR_SCALE_RANGE
    )

    de_indices, log2fcs = assign_de_genes(
        rng, CELLTYPE_DE_CONFIG, N_GENES, D_A_OVERLAP_FRAC
    )

    site_effects = make_site_effects(
        rng, CELLTYPE_NAMES, N_GENES, site_names, SITE_RE_SIGMA
    )

    # ── Generate cell-level counts ────────────────────────────────────

    count_blocks = []
    obs_blocks = []
    cell_counter = 0

    for ct in CELLTYPE_NAMES:
        for _, donor_row in donor_meta.iterrows():
            donor_id = donor_row["donor_id"]
            condition = donor_row["condition"]
            site = donor_row["site"]

            # Variable cell count per donor
            n_cells = rng.poisson(MEAN_CELLS_PER_DONOR)
            n_cells = max(n_cells, MIN_CELLS_PER_DONOR)

            # Site RE for this donor's site
            site_shift = site_effects[ct][site].values

            counts = generate_chunk_counts(
                rng=rng,
                n_cells=n_cells,
                base_means=base_means,
                nb_sizes=nb_sizes,
                ct_scale=ct_scales[ct],
                log2fc=log2fcs[ct],
                site_shift=site_shift,
                condition=condition,
                lib_sigma=LIB_SIZE_SIGMA,
            )

            count_blocks.append(sparse.csr_matrix(counts.astype(np.float32)))

            # Cell-level obs
            cell_ids = [f"cell_{cell_counter + j:06d}" for j in range(n_cells)]
            cell_counter += n_cells

            obs_block = pd.DataFrame({
                "cell_id": cell_ids,
                "sample_id": donor_id,
                "celltype": ct,
                "condition": condition,
                "site": site,
            })
            obs_blocks.append(obs_block)

    # ── Assemble AnnData ──────────────────────────────────────────────

    X = sparse.vstack(count_blocks, format="csr")
    obs = pd.concat(obs_blocks, ignore_index=True)
    obs = obs.set_index("cell_id")
    var = pd.DataFrame(index=gene_names)

    adata = ad.AnnData(X=X, obs=obs, var=var)

    # ── Write .h5ad ───────────────────────────────────────────────────

    h5ad_path = outdir / "tutorial_dataset.h5ad"
    adata.write_h5ad(h5ad_path)

    # ── Gene manifest (ground truth) ──────────────────────────────────

    manifest_rows = []
    for g in range(N_GENES):
        row = {"gene": gene_names[g]}
        for ct in CELLTYPE_NAMES:
            row[f"de_{ct}"] = g in de_indices[ct]
            row[f"log2fc_{ct}"] = log2fcs[ct][g]
            row[f"scale_{ct}"] = ct_scales[ct][g]
        manifest_rows.append(row)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(outdir / "gene_manifest.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────

    print(f"AnnData: {adata.shape[0]:,} cells × {adata.shape[1]} genes")
    print(f"  Donors:     {adata.obs['sample_id'].nunique()}")
    print(f"  Sites:      {adata.obs['site'].nunique()}")
    print(f"  Cell types: {adata.obs['celltype'].nunique()}")
    print(f"  Conditions: {adata.obs['condition'].value_counts().to_dict()}")
    print()
    for ct in CELLTYPE_NAMES:
        n_cells = (adata.obs["celltype"] == ct).sum()
        n_de = len(de_indices[ct])
        print(f"  {ct}: {n_cells:,} cells, {n_de} DE genes")
    print()

    # File size
    size_mb = h5ad_path.stat().st_size / (1024 * 1024)
    print(f"Written: {h5ad_path}  ({size_mb:.1f} MB)")
    print(f"         {outdir / 'gene_manifest.csv'}")


if __name__ == "__main__":
    main()
