#!/usr/bin/env python3
"""
preprocess_wells2025.py

Load, clean, filter, and prepare the Wells 2025 T cell AnnData object
for downstream dreampy analysis.

Filtering strategy (two stages):
  1. Regex-based removal of 11 categories of technical artifacts
  2. Intersection with protein coding whitelist

Usage:
  # Edit defaults below and run directly:
  python validation/python/preprocess_wells2025.py

  # Or override paths via CLI:
  python validation/python/preprocess_wells2025.py \
      --input  ~/SBW001/projects/wells2025/data/h5ads/wells2025.h5ad \
      --whitelist ~/dreampy/resources/protein_coding_whitelist.txt \
      --output ~/SBW001/projects/wells2025/data/h5ads/wells2025_filtered.h5ad
"""

import argparse
import re
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path

# ── Defaults (edit these for interactive use) ───────────────────────────────
# Script lives at: dreampy/validation/python/preprocess_wells2025.py
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # dreampy/

DEFAULT_INPUT = str(REPO_ROOT / "validation/data/wells2025/wells2025.h5ad")
DEFAULT_WHITELIST = str(REPO_ROOT / "resources" / "protein_coding_whitelist.txt")
DEFAULT_OUTPUT = str(REPO_ROOT / "validation/data/wells2025/wells2025_filtered.h5ad")

# ── Gene filter definitions ─────────────────────────────────────────────────
# Each category maps to a list of regex patterns applied to uppercased gene names.
# Negative lookaheads preserve biologically meaningful genes within broad families.
GENE_FILTERS = {
    # Canonical histones lack poly(A) tails — presence indicates stressed/dying cells
    # Histone variants (H3F3, H2AFZ, H2AFX) are polyadenylated and kept
    "histones": [r"^HIST"],

    # Immunoglobulin V/D/J segments — clonal variation noise
    # KEEPS constant regions (IGHM, IGHA, IGHG, IGHE, IGHD) as class-switching markers
    # IGHD constant region preserved: only IGHD+digit patterns are removed
    "ig_genes": [r"^IGH[VJ]", r"^IGHD\d", r"^IGK[VJ]", r"^IGL[VJ]", r"^IGLL"],

    # Ribosomal proteins — translation machinery, confounded with technical variation
    "ribosomal": [r"^RPS", r"^RPL", r"^MRPS", r"^MRPL"],

    # Mitochondrial genes — cell stress, dying cells, ambient RNA
    "mitochondrial": [r"^MT-"],

    # Metallothioneins — oxidative stress from tissue dissociation
    "metallothioneins": [r"^MT1", r"^MT2"],

    # TCR adaptive variable segments
    # KEEPS: TRAV1-2 (MAIT), TRAV10 (iNKT), TRAJ18 (iNKT), TRAJ33/20/12 (MAIT)
    # KEEPS: all gamma/delta genes (TRDV*, TRGV*, TRDJ*, TRGJ*, TRDC, TRGC*)
    # KEEPS: constant regions (TRAC, TRBC1, TRBC2)
    "tcr_adaptive_variable": [
        r"^TRAV(?!1-2|10)", r"^TRBV",
        r"^TRAJ(?!18|33|20|12)", r"^TRBJ",
    ],

    # Hemoglobin genes — excludes HBEGF (growth factor) and HBP1 (transcription factor)
    "hemoglobin": [r"^HB(?!EGF|P1)", r"^EPOR$", r"^ALAS2$"],

    # Humanin-like mitochondrial rRNA genes — non-coding QC artifacts
    "mtrnr2l": [r"^MTRNR2L"],

    # Heat shock proteins — dissociation stress artifacts
    "heat_shock": [r"^HSP"],

    # DNAJ co-chaperones — dissociation stress artifacts
    "dnaj_chaperones": [r"^DNAJ"],
}

# ── Age group mapping ───────────────────────────────────────────────────────
# Binary classification: young (0) vs old (1)
# Derived from first token of donor_age column (exact ages and age ranges)
# Each observed category manually assigned — no threshold heuristic
AGE_GROUP_MAP = {
    "20": "0", "20-24": "0", "23": "0", "25": "0", "25-30": "0", "27": "0",
    "33": "0", "35-39": "0",
    "40-44": "1", "48": "1", "50-54": "1", "55-60": "1", "56": "1",
    "64": "1", "67": "1", "68": "1", "70-74": "1", "73": "1", "75": "1",
}

# ── Tissue abbreviation mapping ─────────────────────────────────────────────
# For new column to abbreviate tissue names and remove spaces.
# Each observed category manually assigned
TISSUE_ABBREV_MAP = {
    'bone marrow': 'bma', 'jejunal epithelium': 'jel',
    'jejunum lamina propria': 'jlp', 'lung': 'lng',
    'blood': 'bld', 'thoracic lymph node': 'tln',
    'epithelial lining fluid': 'bal', 'spleen': 'spl',
    'mesenteric lymph node': 'mln', 'inguinal lymph node': 'iln'
    }

def get_anndata_size_gb(ad):
    """Return size of AnnData object in GB."""
    nbytes = 0
    # Main matrix
    if hasattr(ad.X, 'data'):  # sparse matrix
        nbytes += ad.X.data.nbytes + ad.X.indices.nbytes + ad.X.indptr.nbytes
    else:  # dense array
        nbytes += ad.X.nbytes
    
    # Layers
    for layer in ad.layers.values():
        if hasattr(layer, 'data'):
            nbytes += layer.data.nbytes + layer.indices.nbytes + layer.indptr.nbytes
        else:
            nbytes += layer.nbytes
    
    # obsm, varm, obsp, varp, uns
    for key in ad.obsm:
        obj = ad.obsm[key]
        if hasattr(obj, 'nbytes'):
            nbytes += obj.nbytes
        elif hasattr(obj, 'memory_usage'):  # DataFrame
            nbytes += obj.memory_usage(deep=True).sum()
    
    for key in ad.varm:
        obj = ad.varm[key]
        if hasattr(obj, 'nbytes'):
            nbytes += obj.nbytes
        elif hasattr(obj, 'memory_usage'):  # DataFrame
            nbytes += obj.memory_usage(deep=True).sum()
    
    for key in ad.obsp:
        obj = ad.obsp[key]
        if hasattr(obj, 'data'):  # sparse
            nbytes += obj.data.nbytes + obj.indices.nbytes + obj.indptr.nbytes
        elif hasattr(obj, 'nbytes'):
            nbytes += obj.nbytes
    
    for key in ad.varp:
        obj = ad.varp[key]
        if hasattr(obj, 'data'):  # sparse
            nbytes += obj.data.nbytes + obj.indices.nbytes + obj.indptr.nbytes
        elif hasattr(obj, 'nbytes'):
            nbytes += obj.nbytes

    for key in ad.uns:
        obj = ad.uns[key]
        if hasattr(obj, 'data'):  # sparse
            nbytes += obj.data.nbytes + obj.indices.nbytes + obj.indptr.nbytes
        elif hasattr(obj, 'nbytes'):
            nbytes += obj.nbytes
        elif hasattr(obj, 'memory_usage'):  # DataFrame
            nbytes += obj.memory_usage(deep=True).sum()
    
    nbytes += ad.obs.memory_usage(deep=True).sum()
    nbytes += ad.var.memory_usage(deep=True).sum()
    
    return nbytes / (1024**3)

def filter_genes(gene_list, filters):
    """
    Apply regex-based gene filters and report exclusions by category.

    Parameters
    ----------
    gene_list : array-like of str
        Gene names (will be uppercased internally).
    filters : dict
        Category name -> list of regex patterns.

    Returns
    -------
    genes_kept : list of str
        Uppercased gene names that passed all filters.
    report : dict
        Per-category counts, excluded gene lists, and summary statistics.
    """
    gene_list_upper = [str(g).upper() for g in gene_list]

    excluded_by_category = {}
    for category, patterns in filters.items():
        hits = set()
        for gene in gene_list_upper:
            if any(re.match(pat, gene) for pat in patterns):
                hits.add(gene)
        excluded_by_category[category] = sorted(hits)

    all_excluded = set()
    for genes in excluded_by_category.values():
        all_excluded.update(genes)

    genes_kept = [g for g in gene_list_upper if g not in all_excluded]

    report = {}
    for category, genes in excluded_by_category.items():
        report[category] = {"count": len(genes), "genes": genes}
    report["total_excluded"] = len(all_excluded)
    report["genes_retained"] = len(genes_kept)
    report["pct_retained"] = 100 * len(genes_kept) / len(gene_list_upper) if gene_list_upper else 0.0

    return genes_kept, report

def load_whitelist(whitelist_path):
    """Load GENCODE whitelist (TSV, gene names in column index 1)."""
    path = Path(whitelist_path).expanduser()
    if not path.exists():
        print(f"ERROR: Whitelist file not found: {path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading whitelist from: {path}")
    data = np.loadtxt(path, delimiter="\t", dtype="str")
    return data[:, 1].tolist()

def main(input_path, whitelist_path, output_path):
    # ── Load ────────────────────────────────────────────────────────────────
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path}")
    ad = sc.read_h5ad(input_path)
    print(f"Loaded. Shape: {ad.shape}. Size: {get_anndata_size_gb(ad):.3f} GB")

    # ── Clean & downcast ────────────────────────────────────────────────────
    if ad.raw is None:
        print("WARNING: .raw slot is empty - using .X as-is")
    else:
        print("Replacing .X with raw counts from .raw...")
        ad.X = ad.raw.X.copy()
        del ad.raw
        print(f"    .raw consumed. Size: {get_anndata_size_gb(ad):.3f} GB")

    if ad.uns:
        ad.uns.clear()
        print(f"    .uns removed. Size: {get_anndata_size_gb(ad):.3f} GB")

    print("Downcasting .X to uint16...")
    if hasattr(ad.X, 'data'):
        max_val = ad.X.data.max()
    else:
        max_val = ad.X.max()
    if max_val > 65535:
        print(f"WARNING: max count {max_val} exceeds uint16 range")
    else:
        ad.X = ad.X.astype("uint16")
    print(f" Downcast complete. Size: {get_anndata_size_gb(ad):.3f} GB")

    # ── Defensive uppercase on feature names ────────────────────────────────
    ad.var["feature_name"] = ad.var["feature_name"].str.upper()

    # ── Stage 1: Regex artifact filtering ───────────────────────────────────
    genes = ad.var['feature_name'].values
    genes_after_regex, report = filter_genes(genes, GENE_FILTERS)

    print(f"\n Stage 1: Regex artifact filtering")
    print("=" * 40)
    print(f"Total genes input: {len(genes)}")
    print(f"\nExclusions by category:")
    for category in GENE_FILTERS:
        info = report[category]
        print(f"    {category}: {info['count']} genes")
        if info["count"] > 0:
            examples = ", ".join(info["genes"][:5])
            if info["count"] > 5:
                examples += ", ..."
            print(f"    Examples: {examples}")
    print(f"\n Total excluded: {report['total_excluded']}")
    print(f"Genes retained: {report['genes_retained']}")
    print(f" Percent retained: {report['pct_retained']:.2f}%")

    # ── Stage 2: Whitelist intersection ─────────────────────────────────────
    print(f"\nStage 2: Whitelist intersection")
    print("=" * 40)
    whitelist = load_whitelist(whitelist_path)
    genes_after_regex_set = set(genes_after_regex)
    whitelist_upper = {g.upper() for g in whitelist}
    genes_final = sorted(genes_after_regex_set & whitelist_upper)
    genes_whitelist_excluded = genes_after_regex_set - whitelist_upper

    print(f"Whitelist genes: {len(whitelist_upper)}")
    print(f"Excluded by whitelist: {len(genes_whitelist_excluded)}")
    print(f"Genes retained: {len(genes_final)}")
    n_after_regex = len(genes_after_regex)
    pct = 100 * len(genes_final) / n_after_regex if n_after_regex else 0.0
    print(f"Percent retained (of stage 1 output): {pct:.2f}%")

    # ── Apply gene filter to AnnData ────────────────────────────────────────
    mask = ad.var["feature_name"].isin(genes_final)
    ad = ad[:, mask].copy()
    ad.var["ensembl_id"] = ad.var.index
    ad.var = ad.var.set_index("feature_name")
    print(f"\nFiltered object: {ad.shape[0]} cells x {ad.shape[1]} genes")
    print(f"Size: {get_anndata_size_gb(ad):.3f} GB")

    # ── Age group metadata ──────────────────────────────────────────────────
    print(f"\nCreating age_groups column...")
    ad.obs["age_groups"] = ad.obs["donor_age"].str.split(" ", expand=True)[0]
    ad.obs["age_groups"] = ad.obs["age_groups"].replace(AGE_GROUP_MAP)

    # Check for unmapped values
    unmapped = set(ad.obs["age_groups"].unique()) - {"0", "1"}
    if unmapped:
        print(
            f"WARNING: {len(unmapped)} unmapped age_groups values: {sorted(unmapped)}",
            file=sys.stderr,
        )
        print("  These donors will have non-binary age_groups — check AGE_GROUP_MAP")
    else:
        counts = ad.obs["age_groups"].value_counts()
        print(f"    Young (0): {counts.get('0', 0)} cells, Old (1): {counts.get('1', 0)} cells")

    # ── Tissue abbreviation metadata ──────────────────────────────────────────
    print(f"\nCreating tissue_abbrev column...")
    ad.obs['tissue_abbrev'] = ad.obs['tissue'].astype(str)
    ad.obs['tissue_abbrev'] = ad.obs['tissue_abbrev'].replace(TISSUE_ABBREV_MAP)

    # Check for unmapped values
    unmapped = set(ad.obs["tissue_abbrev"].unique()) - set(TISSUE_ABBREV_MAP.values())
    if unmapped:
        print(
            f"WARNING: {len(unmapped)} unmapped tissue_abbrev values: {sorted(unmapped)}",
            file=sys.stderr,
        )
        print("  These tissues will not have abbreviations — check TISSUE_ABBREV_MAP")
    else:
        print(" Tissue abbreviations successfully mapped")

    # ── Donor + tissue metadata ───────────────────────────────────────────────
    print(f"\nCreating donor_id + tissue_abbrev column...")
    column1 = "donor_id"
    column2 = "tissue_abbrev"
    ad.obs[f"{column1}_{column2}"] = ad.obs[column1].astype(str) + "_" + ad.obs[column2].astype(str)

    if f"{column1}_{column2}" in ad.obs.keys():
        print(" donor_id_tissue_abbrev column successfully added")
    else:
        print(
            f"WARNING: failed to add donor_id_tissue_abbrev column, check object",
            file=sys.stderr,
        )
    
    ad.var_names_make_unique()
    ad._sanitize()
    print(f"Size: {get_anndata_size_gb(ad):.3f} GB")

    # ── Save ────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {output_path}")
    ad.write_h5ad(output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter and prepare Wells 2025 T cell object for dreampy."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to input h5ad")
    parser.add_argument("--whitelist", default=DEFAULT_WHITELIST, help="Path to protein coding whitelist")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to output h5ad")
    args = parser.parse_args()

    main(args.input, args.whitelist, args.output)