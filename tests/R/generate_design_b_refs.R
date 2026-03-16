#!/usr/bin/env Rscript
# generate_design_b_refs.R
# ------------------------
# Runs dreamlet's processAssays + dreamlet pipeline on Design B synthetic data
# using the scaffold+swap approach, and exports all intermediate and final
# values as CSV reference files.
#
# Scaffold+swap:
#   1. Create a fake cell-level SCE with trivial counts.
#   2. Call aggregateToPseudoBulk() to get a correctly-structured pseudobulk SCE.
#   3. Swap in the designed count matrix from counts.csv.
#   4. Run processAssays() → dreamlet() → extract results.
#
# Usage:
#   Rscript generate_design_b_refs.R [DATADIR]
#
#   DATADIR defaults to ../data/synthetic/design_b
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
DATADIR <- if (length(args) >= 1) args[1] else file.path("..", "data", "synthetic", "design_b")

# ── Load designed data ─────────────────────────────────────────────────────
counts <- as.matrix(read.csv(file.path(DATADIR, "counts.csv"), row.names = 1,
                                check.names = FALSE))
meta <- read.csv(file.path(DATADIR, "metadata.csv"), stringsAsFactors = FALSE)
rownames(meta) <- meta$sample_id

stopifnot(identical(colnames(counts), meta$sample_id))
cat("Design B: loaded", nrow(counts), "genes x", ncol(counts), "samples\n")

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

# Cell-level metadata
cell_meta <- data.frame(
    cell_id = paste0("cell_", seq_len(n_total_cells)),
    sample_id = rep(sample_ids, each = n_cells_per_sample),
    cluster_id = rep("celltype_A", n_total_cells),
    stringsAsFactors = FALSE
)

#Propagate sample-level covariates to cells
cell_meta$condition <- meta$condition[match(cell_meta$sample_id, meta$sample_id)]
cell_meta$collection_site <- meta$collection_site[match(cell_meta$sample_id, meta$sample_id)]

# Trivial cell-level count matrix: n_genes x n_total_cells
# These don't matter, they are replace after aggregation
dummy_counts <- matrix(
    rpois(n_genes * n_total_cells, lambda = 10),
    nrow = n_genes, ncol = n_total_cells
)
rownames(dummy_counts) <- gene_names
colnames(dummy_counts) <- cell_meta$cell_id

# Construct SingleCellExperiment

sce <- SingleCellExperiment(
    assays = list(counts = dummy_counts),
    colData = DataFrame(cell_meta, row.names = cell_meta$cell_id)
)

cat("   Scaffold SCE:", ncol(sce), "cells,", nrow(sce), "genes\n")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Pseudobulk aggregation (builds correct internal structures)
# ══════════════════════════════════════════════════════════════════════════

pb <- aggregateToPseudoBulk(
    sce,
    assay = "counts",
    cluster_id = "cluster_id",
    sample_id = "sample_id",
    verbose = FALSE
)

cat("   Pseudobulk SCE:", ncol(pb), "samples,",
        length(assayNames(pb)), "assay(s):", paste(assayNames(pb), collapse = ", "), "\n")

# Verify smaple order matches metadata
pb_samples <- colnames(assay(pb, "celltype_A"))
stopifnot(all(sample_ids %in% pb_samples))

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Swap in the designed count matrix
# ══════════════════════════════════════════════════════════════════════════

# Reorder the designed counts to match pseudobulk sample order
designed_counts <- counts[, pb_samples, drop = FALSE]
assay(pb, "celltype_A") <- designed_counts

cat("   Count matrix swapped. Verifying dimensions:",
        nrow(assay(pb, "celltype_A")), "x", ncol(assay(pb, "celltype_A")), "\n")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: processAssays (filtering + TMM + voom weights)
# ══════════════════════════════════════════════════════════════════════════

formula_str <- ~ condition + (1 | collection_site)

res.proc <- processAssays(pb, formula_str, quiet = FALSE,
                            min.cells = 1, # don't filter by cell count
                            min.count = 5, min.samples = 4, min.prop = 0.4)

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
# processAssays runs calcNormFactors internally. We recover the effective
# library sizes from the voom EList targets
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
    stop("dreamlet returned NULL for celltype_A - modle fitting failed entirely")
}

coef_names <- colnames(fit$coefficients)
cat("   dreamlet fitted:", nrow(fit$coefficients), "genes,",
        length(coef_names), "coefficients:", paste(coef_names, collapse = ", "), "\n")

# -- Extract lmer-specific quantities --

lmer_df <- data.frame(
    gene = rownames(fit$coefficients),
    sigma = fit$sigma,
    stringsAsFactors = FALSE
)

# Coefficients 
for (cn in coef_names) {
    lmer_df[[paste0("coef_", cn)]] <- fit$coefficients[, cn]
}

# stdev.unscaled
for (cn in coef_names) {
    lmer_df[[paste0("stdev_unscaled_", cn)]] <- fit$stdev.unscaled[, cn]
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

# Per-scalar df quantities
# rdf: scalar residual df per gene (trace-based, fed to eBayes)
# edf: effective number of parameters per gene (rdf ≈ n - edf)
if (!is.null(fit$rdf)) {
    lmer_df$rdf <- fit$rdf
} else {
  cat("  WARNING: fit$rdf is NULL.\n")
}

if (!is.null(fit$edf)) {
    lmer_df$edf <- fit$edf
} else {
  cat("  WARNING: fit$edf is NULL.\n")
}

write.csv(lmer_df, file.path(DATADIR, "ref_lmer.csv"), row.names = FALSE)

# ── eBayes results ──
# dreamlet applies eBayes internally. Results are in the fit object.
ebayes_df <- data.frame(
    gene = rownames(fit$coefficients),
    s2_prior = fit$s2.prior,
    df_prior = fit$df.prior,
    s2_post = fit$s2.post,
    stringsAsFactors = FALSE
)

for (cn in coef_names) {
  ebayes_df[[paste0("df_total_", cn)]] <- fit$df.total[, cn]
}

for (cn in coef_names) {
  ebayes_df[[paste0("t_", cn)]]     <- fit$t[, cn]
  ebayes_df[[paste0("p_", cn)]]     <- fit$p.value[, cn]
  ebayes_df[[paste0("b_", cn)]]     <- fit$lods[, cn]
  ebayes_df[[paste0("adj_p_", cn)]] <- p.adjust(fit$p.value[, cn], method = "BH")
}

write.csv(ebayes_df, file.path(DATADIR, "ref_ebayes.csv"), row.names = FALSE)

# ── Satterthwaite df ──
# fit$df.residual: genes × coefficients matrix. Per-coefficient Satterthwaite
#   denominator df BEFORE eBayes adds df.prior. This is what pydreamlet's
#   Satterthwaite computation must match.
# fit$rdf: per-gene scalar. Trace-based residual df used as input to eBayes.
#   Already exported in ref_lmer.csv, but included here for completeness.

satt_df <- data.frame(
  gene = rownames(fit$coefficients),
  rdf  = fit$rdf,
  stringsAsFactors = FALSE
)

# Per-coefficient Satterthwaite df (pre-eBayes)
for (cn in coef_names) {
  satt_df[[paste0("df_residual_", cn)]] <- fit$df.residual[, cn]
}

write.csv(satt_df, file.path(DATADIR, "ref_satterthwaite.csv"), row.names = FALSE)

cat("\nDesign B references written to:", normalizePath(DATADIR), "\n")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: Kenward-Roger pass (same data, different ddf method)
# ══════════════════════════════════════════════════════════════════════════
# KR changes:
#   - stdev.unscaled (bias-corrected vcov → slightly larger)
#   - df.residual (KR denominator df, different from Satterthwaite)
#   - everything downstream: t, p, B, adj_p, df.total
# KR does NOT change:
#   - coefficients (point estimates)
#   - sigma (residual SD)
#   - varComp/theta (variance component estimates)
#
# Verification: if KR is engaged, stdev.unscaled should differ from the
# Satterthwaite pass. If identical, KR is not being applied.

cat("\n── Kenward-Roger pass ──\n")

res.dl.kr <- dreamlet(res.proc, formula_str, ddf = "Kenward-Roger")

fit_kr <- res.dl.kr[["celltype_A"]]

if (is.null(fit_kr)) {
    stop("dreamlet (KR) returned NULL for celltype_A")
}

coef_names_kr <- colnames(fit_kr$coefficients)
cat("   dreamlet (KR) fitted:", nrow(fit_kr$coefficients), "genes,",
        length(coef_names_kr), "coefficients\n")

# ── Sanity check: verify KR is actually engaged ──
# Compare stdev.unscaled for first gene between Satterthwaite and KR
satt_su <- fit$stdev.unscaled[1, ]
kr_su   <- fit_kr$stdev.unscaled[1, ]
if (identical(satt_su, kr_su)) {
    cat("  WARNING: stdev.unscaled identical for gene 1 — KR may not be engaged!\n")
} else {
    cat("  KR confirmed: stdev.unscaled differs from Satterthwaite for gene 1\n")
    cat("    Satterthwaite:", round(satt_su, 6), "\n")
    cat("    KR:           ", round(kr_su, 6), "\n")
}

# ── Export ref_lmer_kr.csv ──
# Only the quantities that differ from Satterthwaite: stdev.unscaled, rdf, edf.
# Coefficients and sigma are identical — we include them anyway for self-contained
# test loading, but the test can assert they match the Satterthwaite version.
lmer_kr_df <- data.frame(
    gene  = rownames(fit_kr$coefficients),
    sigma = fit_kr$sigma,
    stringsAsFactors = FALSE
)

for (cn in coef_names_kr) {
    lmer_kr_df[[paste0("coef_", cn)]] <- fit_kr$coefficients[, cn]
}

for (cn in coef_names_kr) {
    lmer_kr_df[[paste0("stdev_unscaled_", cn)]] <- fit_kr$stdev.unscaled[, cn]
}

if (!is.null(fit_kr$rdf)) {
    lmer_kr_df$rdf <- fit_kr$rdf
}

if (!is.null(fit_kr$edf)) {
    lmer_kr_df$edf <- fit_kr$edf
}

write.csv(lmer_kr_df, file.path(DATADIR, "ref_lmer_kr.csv"), row.names = FALSE)

# ── Export ref_ebayes_kr.csv ──
ebayes_kr_df <- data.frame(
    gene     = rownames(fit_kr$coefficients),
    s2_prior = fit_kr$s2.prior,
    df_prior = fit_kr$df.prior,
    s2_post  = fit_kr$s2.post,
    stringsAsFactors = FALSE
)

for (cn in coef_names_kr) {
    ebayes_kr_df[[paste0("df_total_", cn)]] <- fit_kr$df.total[, cn]
}

for (cn in coef_names_kr) {
    ebayes_kr_df[[paste0("t_", cn)]]     <- fit_kr$t[, cn]
    ebayes_kr_df[[paste0("p_", cn)]]     <- fit_kr$p.value[, cn]
    ebayes_kr_df[[paste0("b_", cn)]]     <- fit_kr$lods[, cn]
    ebayes_kr_df[[paste0("adj_p_", cn)]] <- p.adjust(fit_kr$p.value[, cn], method = "BH")
}

write.csv(ebayes_kr_df, file.path(DATADIR, "ref_ebayes_kr.csv"), row.names = FALSE)

# ── Export ref_kr_df.csv ──
kr_df <- data.frame(
    gene = rownames(fit_kr$coefficients),
    rdf  = fit_kr$rdf,
    stringsAsFactors = FALSE
)

for (cn in coef_names_kr) {
    kr_df[[paste0("df_residual_", cn)]] <- fit_kr$df.residual[, cn]
}

write.csv(kr_df, file.path(DATADIR, "ref_kr_df.csv"), row.names = FALSE)

cat("Design B (Kenward-Roger) references written.\n")

# ── Environment snapshot ──
cat("\n── R session info ──\n")
cat("R version:", R.version.string, "\n")
for (pkg in c("dreamlet", "variancePartition", "limma", "edgeR", 
                "lme4", "lmerTest", "SingleCellExperiment")) {
  v <- tryCatch(as.character(packageVersion(pkg)), error = function(e) "not installed")
  cat(sprintf("  %-25s %s\n", pkg, v))
}