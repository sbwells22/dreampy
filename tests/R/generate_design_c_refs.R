#!/usr/bin/env Rscript
# generate_design_c_refs.R
# ------------------------
# Runs dreamlet's processAssays + dreamlet pipeline on Design C synthetic data
# (multiple random effects) using the scaffold+swap approach.
#
# Design C: ~ condition + (1|collection_site) + (1|batch)
# 30 samples, 5 sites × 3 batches (crossed), 300 genes.
#
# Usage:
#   Rscript generate_design_c_refs.R [DATADIR]
#
#   DATADIR defaults to ../data/synthetic/design_c
#
# Prerequisites:
#   - generate_synthetic_data.R must have been run first.
#   - R packages: dreamlet, variancePartition, SingleCellExperiment, edgeR, limma, lme4

suppressPackageStartupMessages({
  library(dreamlet)
  library(variancePartition)
  library(SingleCellExperiment)
  library(edgeR)
  library(limma)
  library(lme4)
})

args <- commandArgs(trailingOnly = TRUE)
DATADIR <- if (length(args) >= 1) args[1] else file.path("..", "data", "synthetic", "design_c")

# ── Load designed data ─────────────────────────────────────────────────────
counts <- as.matrix(read.csv(file.path(DATADIR, "counts.csv"), row.names = 1,
                                check.names = FALSE))
meta <- read.csv(file.path(DATADIR, "metadata.csv"), stringsAsFactors = FALSE)
rownames(meta) <- meta$sample_id

stopifnot(identical(colnames(counts), meta$sample_id))
cat("Design C: loaded", nrow(counts), "genes x", ncol(counts), "samples\n")

n_genes <- nrow(counts)
n_samples <- ncol(counts)
gene_names <- rownames(counts)
sample_ids <- meta$sample_id

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Build fake cell-level SCE scaffold
# ══════════════════════════════════════════════════════════════════════════

set.seed(11)

n_cells_per_sample <- 100
n_total_cells <- n_samples * n_cells_per_sample

cell_meta <- data.frame(
    cell_id = paste0("cell_", seq_len(n_total_cells)),
    sample_id = rep(sample_ids, each = n_cells_per_sample),
    cluster_id = rep("celltype_A", n_total_cells),
    stringsAsFactors = FALSE
)

# Propagate sample-level covariates to cells
cell_meta$condition <- meta$condition[match(cell_meta$sample_id, meta$sample_id)]
cell_meta$collection_site <- meta$collection_site[match(cell_meta$sample_id, meta$sample_id)]
cell_meta$batch <- meta$batch[match(cell_meta$sample_id, meta$sample_id)]

dummy_counts <- matrix(
    rpois(n_genes * n_total_cells, lambda = 10),
    nrow = n_genes, ncol = n_total_cells
)
rownames(dummy_counts) <- gene_names
colnames(dummy_counts) <- cell_meta$cell_id

sce <- SingleCellExperiment(
    assays = list(counts = dummy_counts),
    colData = DataFrame(cell_meta, row.names = cell_meta$cell_id)
)

cat("   Scaffold SCE:", ncol(sce), "cells,", nrow(sce), "genes\n")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Pseudobulk aggregation
# ══════════════════════════════════════════════════════════════════════════

pb <- aggregateToPseudoBulk(
    sce,
    assay = "counts",
    cluster_id = "cluster_id",
    sample_id = "sample_id",
    verbose = FALSE
)

cat("    Pseudobulk SCE:", ncol(pb), "samples,",
    length(assayNames(pb)), "assay(s):", paste(assayNames(pb), collapse = ", "), "\n")

pb_samples <- colnames(assay(pb, "celltype_A"))
stopifnot(all(sample_ids %in% pb_samples))

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Swap in the designed count matrix
# ══════════════════════════════════════════════════════════════════════════

designed_counts <- counts[, pb_samples, drop = FALSE]
assay(pb, "celltype_A") <- designed_counts

cat("   Count matrix swapped. Dimensions:",
    nrow(assay(pb, "celltype_A")), "x", ncol(assay(pb, "celltype_A")), "\n")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: processAssays
# ══════════════════════════════════════════════════════════════════════════

formula_str <- ~ condition + (1 | collection_site) + (1 | batch)

res.proc <- processAssays(pb, formula_str, quite = FALSE,
                            min.cells = 1,
                            min.count = 5, min.samples = 4, min.prop = 0.4, span=0.5)

kept_genes <- rownames(res.proc$celltype_A)
cat("   processAssays:", length(kept_genes), "genes retained out of", n_genes, "\n")

# -- Export filter mask --
filter_mask <- data.frame(
    gene = gene_names,
    keep = gene_names %in% kept_genes,
    stringsAsFactors = FALSE
)
write.csv(filter_mask, file.path(DATADIR, "ref_filter_mask.csv"), row.names = FALSE)

# -- Export TMM factors --
tmm_factors <- data.frame(
    sample_id = colnames(res.proc$celltype_A),
    effective_lib_size = res.proc$celltype_A$targets$lib.size,
    norm_factors = res.proc$celltype_A$targets$norm.factors,
    stringsAsFactors = FALSE
)
write.csv(tmm_factors, file.path(DATADIR, "ref_tmm_factors.csv"), row.names = FALSE)

# -- Export log2CPM and voom weights --
write.csv(res.proc$celltype_A$E, file.path(DATADIR, "ref_log2cpm.csv"), row.names = TRUE)
colnames(res.proc$celltype_A$weights) <- colnames(res.proc$celltype_A$E)
rownames(res.proc$celltype_A$weights) <- rownames(res.proc$celltype_A$E)
write.csv(res.proc$celltype_A$weights, file.path(DATADIR, "ref_voom_weights.csv"), row.names = TRUE)

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: dreamlet (mixed model fitting + eBayes)
# ══════════════════════════════════════════════════════════════════════════

res.dl <- dreamlet(res.proc, formula_str)

fit <- res.dl[["celltype_A"]]

if (is.null(fit)) {
    stop("dreamlet returned NULL for celltype_A - mdoel fitting failed entirely")
}

coef_names <- colnames(fit$coefficients)
cat("   dreamlet fitted:", nrow(fit$coefficients), "genes,",
    length(coef_names), "coefficients", paste(coef_names, collapse = ", "), "\n")

# -- Extract lmer quantities --
lmer_df <- data.frame(
    gene = rownames(fit$coefficients),
    sigma = fit$sigma,
    stringsAsFactors = FALSE
)

for (cn in coef_names) {
    lmer_df[[paste0("coef_", cn)]] <- fit$coefficients[,cn]
}

for (cn in coef_names) {
    lmer_df[[paste0("stdev_unscaled_", cn)]] <- fit$stdev.unscaled[,cn]
}

# Variance components: stored as S4 attribute, not in .Data list
# attr(fit, "varComp") is a data.frame: genes × RE components (+ residual)
vc <- attr(fit, "varComp")
if (!is.null(vc)) {
  cat("  varComp columns:", paste(colnames(vc), collapse = ", "), "\n")
  for (cn in colnames(vc)) {
    lmer_df[[paste0("varComp_", cn)]] <- vc[, cn]
  }
} else {
  cat("  WARNING: attr(fit, 'varComp') is NULL. No variance components found.\n")
}

# Per-gene scalar df quantities
if (!is.null(fit$rdf)) {
    lmer_df$rdf <- fit$rdf
} else {
    cat("   WARNING: fit$rdf is NULL.\n")
}

if (!is.null(fit$edf)) {
    lmer_df$edf <- fit$edf
} else {
    cat("   WARNING: fit$edf is NULL.\n")
}

write.csv(lmer_df, file.path(DATADIR, "ref_lmer.csv"), row.names = FALSE)

# -- eBayes results -- 
ebayes_df <- data.frame(
    gene = rownames(fit$coefficients),
    s2_prior = fit$s2.prior,
    df_prior = fit$df.prior,
    s2_post = fit$s2.post,
    stringsAsFactors = FALSE
)

# df.total is a genes x coefficients matrix (Satterthwaite df + df.prior)
for (cn in coef_names) {
    ebayes_df[[paste0("df_total_", cn)]] <- fit$df.total[,cn]
}

for (cn in coef_names) {
    ebayes_df[[paste0("t_", cn)]] <- fit$t[,cn]
    ebayes_df[[paste0("p_", cn)]] <- fit$p.value[,cn]
    ebayes_df[[paste0("b_", cn)]] <- fit$lods[,cn]
    ebayes_df[[paste0("adj_p_", cn)]] <- p.adjust(fit$p.value[,cn], method = "BH")
}

write.csv(ebayes_df, file.path(DATADIR, "ref_ebayes.csv"), row.names = FALSE)

# ── Satterthwaite df ──
# fit$df.residual: genes × coefficients matrix. Per-coefficient Satterthwaite
#   denominator df BEFORE eBayes adds df.prior.
# fit$rdf: per-gene scalar (already in ref_lmer.csv).

satt_df <- data.frame(
  gene = rownames(fit$coefficients),
  rdf  = fit$rdf,
  stringsAsFactors = FALSE
)

for (cn in coef_names) {
  satt_df[[paste0("df_residual_", cn)]] <- fit$df.residual[, cn]
}

write.csv(satt_df, file.path(DATADIR, "ref_satterthwaite.csv"), row.names = FALSE)

cat("\nDesign C references written to:", normalizePath(DATADIR), "\n")

# ── Environment snapshot ──
cat("\n── R session info ──\n")
cat("R version:", R.version.string, "\n")
for (pkg in c("dreamlet", "variancePartition", "limma", "edgeR", 
                "lme4", "lmerTest", "SingleCellExperiment")) {
  v <- tryCatch(as.character(packageVersion(pkg)), error = function(e) "not installed")
  cat(sprintf("  %-25s %s\n", pkg, v))
}