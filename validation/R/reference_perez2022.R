#!/usr/bin/env Rscript
# reference_perez2022.R
# ------------------------
# Runs dreamlet's processAssays + dreamlet pipeline on the Perez SLE data
# and exports all intermediate and final values as CSV reference files.
#
# Usage:
#   Rscript reference_perez2022.R [H5ADDIR] [DATADIR]
#
#   H5ADDIR must be defined
#   DATADIR defaults to ../data/perez2022
#
# Prerequisites:
#   - preprocess_perez2022.py must have been run first.
#   - R packages: dreamlet, variancePartition, SingleCellExperiment, edgeR, 
#     zellkonverter, limma, lme4, BiocParallel

suppressPackageStartupMessages({
  library(dreamlet)
  library(variancePartition)
  library(SingleCellExperiment)
  library(edgeR)
  library(limma)
  library(lme4)
  library(zellkonverter)
  library(BiocParallel)
})

set.seed(11)

args <- commandArgs(trailingOnly = TRUE)
script_dir <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(trailingOnly=FALSE), value=TRUE))))
H5ADDIR <- if (length(args) >= 1) args[1] else file.path(script_dir, "..", "data", "perez2022")
DATADIR <- if (length(args) >= 2) args[2] else file.path(script_dir, "..", "data", "perez2022")

# ── Create output directory if needed ─────────────────────────────────────
dir.create(DATADIR, recursive = TRUE, showWarnings = FALSE)

# ── Parallelization ───────────────────────────────────────────────────────
ncores <- parallel::detectCores()
cat("Using", ncores, "Cores (MulticoreParam)\n")
param <- MulticoreParam(ncores)

# ── Load filtered data ─────────────────────────────────────────────────────
sce = readH5AD(file.path(H5ADDIR, "perez2022_filtered.h5ad"), use_hdf5=TRUE, verbose=FALSE)
gene_names <- rownames(sce)
n_genes <- length(gene_names)
cat("Loaded perez2022_filtered.h5ad:", n_genes, "genes x", ncol(sce), "cells\n")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Pseudobulk aggregation
# ══════════════════════════════════════════════════════════════════════════

pb <- aggregateToPseudoBulk(
    sce,
    assay = "X",
    cluster_id = "author_cell_type",
    sample_id = "sample_id",
    verbose = FALSE
)

cat("    Pseudobulk SCE:", ncol(pb), "samples,",
    length(assayNames(pb)), "assay(s):", paste(assayNames(pb), collapse = ", "), "\n")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: processAssays
# ══════════════════════════════════════════════════════════════════════════

formula_str <- ~ sle + (1 | donor_id) + (1 | Processing_Cohort)

t0 <- proc.time()
res.proc <- processAssays(pb, formula_str, quiet = FALSE,
                            min.cells = 20, span = 0.5,
                            min.count = 5, min.samples = 4, min.prop = 0.4,
                            BPPARAM = param)
time_processAssays <- (proc.time() - t0)[3]
cat("processAssays time:", time_processAssays, "sec\n")


for (i in assayNames(res.proc)) {
  kept_genes <- rownames(res.proc[[i]])
  cat("   processAssays:", length(kept_genes), "genes retained out of", n_genes, "\n")
  
  # -- Export filter mask --
  filter_mask <- data.frame(
      gene = gene_names,
      keep = gene_names %in% kept_genes,
      stringsAsFactors = FALSE
  )
  write.csv(filter_mask, file.path(DATADIR, paste0(i, "_ref_filter_mask.csv")), row.names = FALSE)
  
  # -- Export TMM factors --
  tmm_factors <- data.frame(
      sample_id = colnames(res.proc[[i]]),
      effective_lib_size = res.proc[[i]]$targets$lib.size,
      norm_factors = res.proc[[i]]$targets$norm.factors,
      stringsAsFactors = FALSE
  )
  write.csv(tmm_factors, file.path(DATADIR, paste0(i, "_ref_tmm_factors.csv")), row.names = FALSE)
  
  # -- Export log2CPM and voom weights --
  write.csv(res.proc[[i]]$E, file.path(DATADIR, paste0(i, "_ref_log2cpm.csv")), row.names = TRUE)
  colnames(res.proc[[i]]$weights) <- colnames(res.proc[[i]]$E)
  rownames(res.proc[[i]]$weights) <- rownames(res.proc[[i]]$E)
  write.csv(res.proc[[i]]$weights, file.path(DATADIR, paste0(i, "_ref_voom_weights.csv")), row.names = TRUE) 
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: dreamlet (mixed model fitting + eBayes)
# ══════════════════════════════════════════════════════════════════════════

t0 <- proc.time()
res.dl <- dreamlet(res.proc, formula_str, ddf="Satterthwaite",
                    BPPARAM = param)
time_dreamlet <- (proc.time() - t0)[3]                    
cat("dreamlet time:", time_dreamlet, "sec\n")

for (i in assayNames(res.dl)) {

  fit <- res.dl[[i]]
  
  if (is.null(fit)) {
      warning("dreamlet returned NULL for '", i, "' - model fitting failed entirely")
      next
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
    cat("  varComp rows:", nrow(vc), " vs coefficients rows:", nrow(fit$coefficients), "\n")
    # Match by gene name — varComp may contain genes that failed fitting
    vc_matched <- vc[rownames(fit$coefficients), , drop = FALSE]
    for (cn in colnames(vc_matched)) {
        lmer_df[[paste0("varComp_", cn)]] <- vc_matched[, cn]
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
  
  write.csv(lmer_df, file.path(DATADIR, paste0(i, "_ref_lmer.csv")), row.names = FALSE)
  
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
  
  write.csv(ebayes_df, file.path(DATADIR, paste0(i, "_ref_ebayes.csv")), row.names = FALSE)
  
  # ── Satterthwaite df ──
  # fit$df.residual: genes × coefficients matrix. Per-coefficient Satterthwaite
  #   denominator df BEFORE eBayes adds df.prior.
  
  satt_df <- data.frame(
    gene = rownames(fit$coefficients),
    stringsAsFactors = FALSE
  )
  
  for (cn in coef_names) {
    satt_df[[paste0("df_residual_", cn)]] <- fit$df.residual[, cn]
  }
  
  write.csv(satt_df, file.path(DATADIR, paste0(i, "_ref_satterthwaite.csv")), row.names = FALSE)

}

cat("\nperez2022 references written to:", normalizePath(DATADIR), "\n")

timestamp <- format(Sys.time(), "%d%b%Y_%H%M%S")
timing_file <- file.path(DATADIR, paste0("R_perez2022_timing_", timestamp, ".txt"))
writeLines(c(
    paste("processAssays:", time_processAssays, "sec"),
    paste("dreamlet:", time_dreamlet, "sec"),
    paste("total:", time_processAssays + time_dreamlet, "sec"),
    paste("cores:", ncores),
    paste("timestamp:", Sys.time())
), timing_file)
cat("Timing written to:", timing_file, "\n")

# ── Environment snapshot ──
cat("\n── R session info ──\n")
cat("R version:", R.version.string, "\n")
for (pkg in c("dreamlet", "variancePartition", "limma", "edgeR", "lme4", 
                "lmerTest", "SingleCellExperiment", "BiocParallel", "zellkonverter")) {
  v <- tryCatch(as.character(packageVersion(pkg)), error = function(e) "not installed")
  cat(sprintf("  %-25s %s\n", pkg, v))
}