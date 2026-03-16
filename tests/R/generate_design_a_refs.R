#!/usr/bin/env Rscript
# generate_design_a_refs.R
# ------------------------
# Runs the dreamlet-equivalent FE pipeline on Design A synthetic data and
# exports all intermediate and final values as CSV reference files.
# Uses voomWithDreamWeights (not limma::voom) to match processOneAssay.
#
# Usage:
#   Rscript generate_design_a_refs.R [DATADIR]
#
#   DATADIR defaults to ../data/synthetic/design_a
#
# Prerequisites:
#   - generate_synthetic_data.R must have been run first.
#   - R packages: edgeR, limma, variancePartition (for voomWithDreamWeights)

suppressPackageStartupMessages({
    library(edgeR)
    library(limma)
    library(variancePartition)
})

args <- commandArgs(trailingOnly = TRUE)
DATADIR <- if (length(args) >= 1 ) args[1] else file.path("..", "data", "synthetic", "design_a")

# ── Load data ──────────────────────────────────────────────────────────────
counts <- as.matrix(read.csv(file.path(DATADIR, "counts.csv"), row.names = 1,
                                check.names = FALSE))
meta <- read.csv(file.path(DATADIR, "metadata.csv"), stringsAsFactors = FALSE)
rownames(meta) <- meta$sample_id

# Sanity: column order must match
stopifnot(identical(colnames(counts), meta$sample_id))

cat("Design A: loaded", nrow(counts), "genes x", ncol(counts), "samples\n")

# ── 1. TMM normalization ──────────────────────────────────────────────────
# processOneAssay defaults for the FE case.
y <- DGEList(counts = counts)

y <- calcNormFactors(y, method = "TMM")

tmm_factors <- data.frame(
    sample_id = colnames(y),
    effective_lib_size = y$samples$lib.size * y$samples$norm.factors,
    norm_factors = y$samples$norm.factors,
    stringsAsFactors = FALSE
)
write.csv(tmm_factors, file.path(DATADIR, "ref_tmm_factors.csv"), row.names = FALSE)

# ── 2. Gene filtering ─────────────────────────────────────────────────────
# Use edgeR::filterByExpr with default-like parameters matching dreamlet's
# filterByExpr needs a design or group to estimate the minimum group size
# We pass the design matrix so it can infer min.n form the smallest group

keep <- filterByExpr(y, min.count = 5, min.prop = 0.4, min.total.count = 15)

filter_mask <- data.frame(
    gene = rownames(counts),
    keep = keep,
    stringsAsFactors = FALSE
)

write.csv(filter_mask, file.path(DATADIR, "ref_filter_mask.csv"), row.names = FALSE)
cat(" filterByExpr:", sum(keep), "genes retained,", sum(!keep), "filtered\n")

# Subset — keep.lib.sizes=TRUE matches DGEList default in processOneAssay
y <- y[keep, , keep.lib.sizes = TRUE]

# ── 3. voomWithDreamWeights ───────────────────────────────────────────────
# processOneAssay passes a matrix of 1s as weights when weightsList is NULL
formula_str <- ~ condition + age
precWeights <- matrix(1, nrow = sum(keep), ncol = ncol(y))
rownames(precWeights) <- rownames(y)
colnames(precWeights) <- colnames(y)

v <- voomWithDreamWeights(y, formula_str, meta,
    weights = precWeights, span = 0.5, save.plot = FALSE, quiet = TRUE)

# v$E is log2 CPM, v$weights is the precision weights matrix
write.csv(v$E, file.path(DATADIR, "ref_log2cpm.csv"), row.names = TRUE)
rownames(v$weights) <- rownames(v$E)
colnames(v$weights) <- colnames(v$E)
write.csv(v$weights, file.path(DATADIR, "ref_voom_weights.csv"), row.names = TRUE)

# ── 4. lmFit ──────────────────────────────────────────────────────────────
fit <- lmFit(v, v$design)

# Extract: coefficients, sigma, stdev.unscaled
# Wide format: one row per gene, coefficient names as the column prefixes
n_genes_kept <- nrow(fit$coefficients)
coef_names <- colnames(fit$coefficients)

lmfit_df <- data.frame(
    gene = rownames(fit$coefficients),
    sigma = fit$sigma,
    stringsAsFactors = FALSE
)

# Add coefficients
for (cn in coef_names) {
    lmfit_df[[paste0("coef_", cn)]] <- fit$coefficients[, cn]
}

# Add stdev.unscaled
for (cn in coef_names) {
    lmfit_df[[paste0("stdev_unscaled_", cn)]] <- fit$stdev.unscaled[, cn]
}

write.csv(lmfit_df, file.path(DATADIR, "ref_lmfit.csv"), row.names = FALSE)
cat("   lmFit: fitted", n_genes_kept, "genes,", length(coef_names), "coefficients\n")

# ── 5. eBayes ─────────────────────────────────────────────────────────────
fit2 <- eBayes(fit)

ebayes_df <- data.frame(
    gene = rownames(fit2$coefficients),
    s2_prior = fit2$s2.prior,
    df_prior = fit2$df.prior,
    s2_post = fit2$s2.post,
    stringsAsFactors = FALSE
)

# Per-coefficient: t, p.value, lods (B)
# df.total is per-gene (scalar per gene, same across coefficients in FE)
ebayes_df$df_total <- fit2$df.total

for (cn in coef_names) {
    ebayes_df[[paste0("t_", cn)]] <- fit2$t[, cn]
    ebayes_df[[paste0("p_", cn)]] <- fit2$p.value[, cn]
    ebayes_df[[paste0("b_", cn)]] <- fit2$lods[, cn]
}

# Adjusted p-values (BH) per coefficient
for (cn in coef_names) {
    ebayes_df[[paste0("adj_p_", cn)]] <- p.adjust(fit2$p.value[, cn], method = "BH")
}

write.csv(ebayes_df, file.path(DATADIR, "ref_ebayes.csv"), row.names = FALSE)
cat("  eBayes: s2.prior =", signif(fit2$s2.prior, 5),
    ", df.prior =", signif(fit2$df.prior, 5), "\n")

cat("Design A references written to:", normalizePath(DATADIR), "\n")