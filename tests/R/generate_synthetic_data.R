#!/usr/bin/env Rscript
# generate_synthetic_data.R
# --------------------
# Generates counts.csv, metadata.csv, and gene_manifest.csv for all three
# synthetic test designs. No statistical pipeline is run here — that is
# handled by the per-design reference scripts.
#
# Usage:
#   Rscript generate_synthetic_data.R [OUTDIR]
#
#   OUTDIR defaults to ../data/synthetic (relative to this script's location)
#
# Seed: 11

# ── Setup ──────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
OUTDIR <- if (length(args) >= 1) args[1] else file.path("..", "data", "synthetic")

for (d in c("design_a", "design_b", "design_c")) {
    dir.create(file.path(OUTDIR, d), recursive = TRUE, showWarnings = FALSE)
}

set.seed(11)

# ── Helpers ────────────────────────────────────────────────────────────────

#' Simulate NB counts for one gene across samples.
#' 
#' @param n_samples Number of samples.
#' @param base_mean Baseline mean (before effects).
#' @param size NB size parameter (dispersion = 1/size).
#' @param lib_factors Per-sample library size multipliers (length n_samples).
#' @param condition Character vector ("case"/"control"), length n_samples.
#' @param log2fc log2 fold-change applied to "case" sample samples. 0 = null gene.
#' @param re_shifts Named list of numeric vectors. Each name is an RE grouping
#'                  variable; each vector is the per-level multiplicative shift
#'                  (on the log scale) already expanded to per-sample.
#'                  Example: list(site = c(0.3, 0.3, -0.1, -0.1, ...))
sim_gene <- function(n_samples, base_mean, size, lib_factors, 
                     condition, log2fc = 0, re_shifts = list()) {
mu <- rep(base_mean, n_samples) * lib_factors

 # Fixed effect: condition
 is_case <- condition == "case"
 mu[is_case] <- mu[is_case] * 2^log2fc

 # Random effects (multiplicative on natural scale)
 for (nm in names(re_shifts)){
    mu <- mu * 2^re_shifts[[nm]]
 }

 mu <- pmax(mu, 1e-6) # guard against zero/negative mu
 rnbinom(n_samples, size = size, mu = mu)
}


# ══════════════════════════════════════════════════════════════════════════
# DESIGN A — Fixed Effects  (~ condition + age)
# ══════════════════════════════════════════════════════════════════════════

n_a <- 20
n_genes_a <- 200

# -- Metadata --
condition_a <- rep(c("case", "control"), each = 10)
age_a <- round(rnorm(n_a, mean = 45, sd = 10))
redundant_a <- ifelse(condition_a == "case", "treated", "untreated")
collinear_a <- 2 * age_a + 5

meta_a <- data.frame(
    sample_id = sprintf("sample_%02d", 1:n_a),
    condition = condition_a,
    age = age_a,
    redundant = redundant_a,
    collinear = collinear_a,
    stringsAsFactors = FALSE
)

# -- Per-gene parameters --
# Total genes hardcoded to 200, the following number of gene 
# modifications sum to that value.
base_means_a <- runif(n_genes_a, 5, 500)
sizes_a <- runif(n_genes_a, 0.5, 5)
lib_factors_a <- exp(rnorm(n_a, 0, 0.3))

# Decide which standard genes (1-180) have a condition effect (~30%)
has_effect_a <- rep(FALSE, 180)
has_effect_a[sample(180, size = round(180 * 0.3))] <- TRUE
log2fc_a <- ifelse(has_effect_a, rnorm(180, 1, 0.3), 0)

# -- Generate count matrix --
counts_a <- matrix(0L, nrow = n_genes_a, ncol = n_a)
rownames(counts_a) <- sprintf("gene_%03d", 1:n_genes_a)
colnames(counts_a) <- meta_a$sample_id

# Genes 1-180: standard NB
for (g in 1:180) {
    counts_a[g, ] <- sim_gene(n_a, base_means_a[g], sizes_a[g],
                                lib_factors_a, condition_a, log2fc_a[g])
}

# Genes 181-185: all-zero
for (g in 181:185) {
    counts_a[g, ] <- 0L
}

# Genes 186-190: very low counts (borderline filtering)
for (g in 186:190) {
    counts_a[g, ] <- 0L
    # spike 1-2 samples with small counts
    spike_idx <- sample(n_a, size = sample(1:2, 1))
    counts_a[g, spike_idx] <- sample(1:3, length(spike_idx), replace = TRUE)
}

# Genes 191-195: moderate counts, no conditon variance (null)
for (g in 191:195) {
    counts_a[g, ] <- sim_gene(n_a, base_means_a[g], sizes_a[g],
                                lib_factors_a, condition_a, log2fc = 0)
}

# Genes 196-200: standard padding
for (g in 196:200) {
    counts_a[g, ] <- sim_gene(n_a, base_means_a[g], sizes_a[g],
                                lib_factors_a, condition_a, log2fc = 0)
}

# -- Gene manifest --
manifest_a <- data.frame(
    gene = rownames(counts_a),
    category = c(
        ifelse(has_effect_a, "standard_effect", "standard_null"), #1-180
        rep("allzero", 5), # 181-185
        rep("borderline_low", 5), # 186-190
        rep("null_moderate", 5), # 191-195
        rep("padding", 5) # 196-200
    ),
    has_condition_effect = c(has_effect_a, rep(FALSE, 20)),
    true_log2fc = c(log2fc_a, rep(0, 20)),
    expected_filtered = c(
        rep(FALSE, 180),
        rep(TRUE, 5), # allzero -> filtered
        rep(TRUE, 5), # borderline -> filtered
        rep(FALSE, 5), # null_moderate -> should pass
        rep(FALSE, 5) # padding -> should pass
    ),
    stringsAsFactors = FALSE
)

# -- Write --
write.csv(counts_a, file.path(OUTDIR, "design_a", "counts.csv"), row.names = TRUE)
write.csv(meta_a, file.path(OUTDIR, "design_a", "metadata.csv"), row.names = FALSE)
write.csv(manifest_a, file.path(OUTDIR, "design_a", "gene_manifest.csv"), row.names = FALSE)
cat("Design A written:", n_genes_a, "genes x", n_a, "samples\n")

# ══════════════════════════════════════════════════════════════════════════
# DESIGN B — Single RE  (~ condition + (1|collection_site))
# ══════════════════════════════════════════════════════════════════════════

n_b <- 24
n_genes_b <- 250
n_sites <- 4
samples_per_site <-6

# -- Metadata --
site_labels <- rep(paste0("site_", LETTERS[1:n_sites]), each = samples_per_site)
condition_b <- rep(c("case", "case", "case", "control", "control", "control"),
                    times = n_sites)

meta_b <- data.frame(
    sample_id = sprintf("sample_%02d", 1:n_b),
    condition = condition_b,
    collection_site = site_labels,
    stringsAsFactors = FALSE
)

# -- Per-gene parameters --
base_means_b <- runif(n_genes_b, 5, 500)
sizes_b <- runif(n_genes_b, 0.5, 5)
lib_factors_b <- exp(rnorm(n_b, 0, 0.3))

has_effect_b <- rep(FALSE, 220)
has_effect_b[sample(220, size = round(220 * 0.3))] <- TRUE
log2fc_b <- ifelse(has_effect_b, rnorm(220, 1, 0.3), 0)

# Standard site RE intercepts (σ_RE = 0.5)
make_site_shifts <- function(n_sites, n_per_site, sigma_re) {
    site_effects <- rnorm(n_sites, 0, sigma_re)
    rep(site_effects, each = n_per_site)
}

# -- Generate count matrix --
counts_b <- matrix(0L, nrow = n_genes_b, ncol = n_b)
rownames(counts_b) <- sprintf("gene_%03d", 1:n_genes_b)
colnames(counts_b) <- meta_b$sample_id

# Genes 1-220: standard with site RE (σ_RE = 0.5)
for (g in 1:220) {
    shifts <- make_site_shifts(n_sites, samples_per_site, 0.5)
    counts_b[g, ] <- sim_gene(n_b, base_means_b[g], sizes_b[g],
                                lib_factors_b, condition_b, log2fc_b[g],
                                re_shifts = list(site = shifts))
}

# Genes 221-225: all-zero
for (g in 221:225) counts_b[g, ] <- 0L

# Genes 226:230: boundary-θ (σ_RE = 0, no site variation)
for (g in 226:230) {
    counts_b[g, ] <- sim_gene(n_b, base_means_b[g], sizes_b[g],
                                lib_factors_b, condition_b, log2fc = 0,
                                re_shifts = list(site = rep(0, n_b)))
}

# Genes 231-235: high site variance (σ_RE = 2.0)
for (g in 231:235) {
    shifts <- make_site_shifts(n_sites, samples_per_site, 2.0)
    counts_b[g, ] <- sim_gene(n_b, base_means_b[g], sizes_b[g],
                                lib_factors_b, condition_b, log2fc = 0,
                                re_shifts = list(site = shifts))
}

# Genes 236-240: site effect correlated with conditon direction
# Sites A,B get positive baseline shift; C,D get negative.
# Combined with the balanced case/control within each site, the mixed model
# should correctly attribute this to site, not condition
for (g in 236:240) {
    # Deterministic site effects correlated with which sites have more "case" signal
    directed_shifts <- rep(c(1.0, 0.8, -0.8, -1.0), each = samples_per_site)
    counts_b[g, ] <- sim_gene(n_b, base_means_b[g], sizes_b[g],
                                lib_factors_b, condition_b,
                                log2fc = rnorm(1, 0.5, 0.1),
                                re_shifts = list(site = directed_shifts))
}

# Genes 241-250:
for (g in 241:250) {
    shifts <- make_site_shifts(n_sites, samples_per_site, 0.5)
    counts_b[g, ] <- sim_gene(n_b, base_means_b[g], sizes_b[g],
                                lib_factors_b, condition_b, log2fc = 0,
                                re_shifts = list(site = shifts))
}

# -- Gene manifest --
manifest_b <- data.frame(
  gene = rownames(counts_b),
  category = c(
    ifelse(has_effect_b, "standard_effect", "standard_null"),
    rep("allzero", 5),
    rep("boundary_theta", 5),
    rep("high_site_var", 5),
    rep("site_condition_corr", 5),
    rep("padding", 10)
  ),
  has_condition_effect = c(has_effect_b, rep(FALSE, 30)),
  true_log2fc = c(log2fc_b, rep(0, 5), rep(0, 5), rep(0, 5), rep(NA, 5), rep(0, 10)),
  true_site_sigma = c(
    rep(0.5, 220), rep(NA, 5), rep(0, 5), rep(2.0, 5), rep(NA, 5), rep(0.5, 10)
  ),
  stringsAsFactors = FALSE
)

write.csv(counts_b, file.path(OUTDIR, "design_b", "counts.csv"), row.names = TRUE)
write.csv(meta_b, file.path(OUTDIR, "design_b", "metadata.csv"), row.names = FALSE)
write.csv(manifest_b, file.path(OUTDIR, "design_b", "gene_manifest.csv"), row.names = FALSE)
cat("Design B written:", n_genes_b, "genes x", n_b, "samples\n")

# ══════════════════════════════════════════════════════════════════════════
# DESIGN C — Multiple RE  (~ condition + (1|collection_site) + (1|batch))
# ══════════════════════════════════════════════════════════════════════════

n_sites_c <- 5
n_batches <- 3
n_per_cell <- 2 # 2 samples per sitexbatch cell
n_c <- n_sites_c * n_batches * n_per_cell # = 30
n_genes_c <- 300

# -- Metadata --
# Crossed design: every site appears in every batch
grid <- expand.grid(
    rep = 1:n_per_cell,
    batch = paste0("batch_", 1:n_batches),
    site = paste0("site_", LETTERS[1:n_sites_c]),
    stringsAsFactors = FALSE
)
# Assigns condition: within each sitexbatch cell, 1 case + 1 control
# (deterministic for reproducibility)
grid$condition <- ifelse(grid$rep == 1, "case", "control")

meta_c <- data.frame(
    sample_id = sprintf("sample_%02d", 1:n_c),
    condition = grid$condition,
    collection_site = grid$site,
    batch = grid$batch,
    stringsAsFactors = FALSE
)

# -- Per-gene parameters --
base_means_c <- runif(n_genes_c, 5, 500)
sizes_c <- runif(n_genes_c, 0.5, 5)
lib_factors_c <- exp(rnorm(n_c, 0, 0.3))

has_effect_c <- rep(FALSE, 260)
has_effect_c[sample(260, size = round(260 * 0.3))] <- TRUE
log2fc_c <- ifelse(has_effect_c, rnorm(260, 1, 0.3), 0)

# Helper: make per-sample RE shifts for the crossed design
make_crossed_shifts <- function(meta, re_col, sigma_re) {
    levels <- unique(meta[[re_col]])
    effects <- rnorm(length(levels), 0, sigma_re)
    names(effects) <- levels
    effects[meta[[re_col]]]
}

# -- Generate count matrix --
counts_c <- matrix(0L, nrow = n_genes_c, ncol = n_c)
rownames(counts_c) <- sprintf("gene_%03d", 1:n_genes_c)
colnames(counts_c) <- meta_c$sample_id

# Genes 1-260: standard with both RE (σ_site = 0.5, σ_batch = 0.3)
for (g in 1:260) {
    site_shifts <- make_crossed_shifts(meta_c, "collection_site", 0.5)
    batch_shifts <- make_crossed_shifts(meta_c, "batch", 0.3)
    counts_c[g, ] <- sim_gene(n_c, base_means_c[g], sizes_c[g],
                                lib_factors_c, meta_c$condition, log2fc_c[g],
                                re_shifts = list(site = site_shifts, batch = batch_shifts))
}

# Genes 261-165: all-zero
for (g in 261:265) counts_c[g, ] <- 0L

# Genes 266-270: both RE -> θ=0 (no site or batch variation)
for (g in 266:270) {
    counts_c[g, ] <- sim_gene(n_c, base_means_c[g], sizes_c[g],
                                lib_factors_c, meta_c$condition, log2fc = 0,
                                re_shifts = list(site = rep(0, n_c),
                                                    batch = rep(0, n_c)))
}

# Genes 271-275: site σ=1.5, batch σ=0 (one informative, one boundary) 
for (g in 271:275) {
    site_shifts <- make_crossed_shifts(meta_c, "collection_site", 1.5)
    counts_c[g, ] <- sim_gene(n_c, base_means_c[g], sizes_c[g],
                                lib_factors_c, meta_c$condition, log2fc = 0,
                                re_shifts = list(site = site_shifts,
                                                    batch = rep(0, n_c)))
}

# Genes 276-280: site σ=0, batch σ=1.5 (opposite pattern)
for (g in 276:280) {
    batch_shifts <- make_crossed_shifts(meta_c, "batch", 1.5)
    counts_c[g, ] <- sim_gene(n_c, base_means_c[g], sizes_c[g],
                                lib_factors_c, meta_c$condition, log2fc = 0,
                                re_shifts = list(site = rep(0, n_c),
                                                    batch = batch_shifts))
}

# Genes 281-285: both RE large (σ_site = 1.5, σ_batch = 1.5)
for (g in 281:285) {
    site_shifts <- make_crossed_shifts(meta_c, "collection_site", 1.5)
    batch_shifts <- make_crossed_shifts(meta_c, "batch", 1.5)
    counts_c[g, ] <- sim_gene(n_c, base_means_c[g], sizes_c[g],
                                lib_factors_c, meta_c$condition, log2fc = 0,
                                re_shifts = list(site = site_shifts, batch = batch_shifts))
}

# Genes 286-300: padding
for (g in 286:300) {
    site_shifts <- make_crossed_shifts(meta_c, "collection_site", 0.5)
    batch_shifts <- make_crossed_shifts(meta_c, "batch", 0.3)
    counts_c[g, ] <- sim_gene(n_c, base_means_c[g], sizes_c[g],
                                lib_factors_c, meta_c$condition, log2fc = 0,
                                re_shifts = list(site = site_shifts, batch = batch_shifts))
}

# ── Gene manifest ──
manifest_c <- data.frame(
  gene = rownames(counts_c),
  category = c(
    ifelse(has_effect_c, "standard_effect", "standard_null"),
    rep("allzero", 5),
    rep("both_boundary_theta", 5),
    rep("site_only", 5),
    rep("batch_only", 5),
    rep("both_high_var", 5),
    rep("padding", 15)
  ),
  has_condition_effect = c(has_effect_c, rep(FALSE, 40)),
  true_log2fc = c(log2fc_c, rep(0, 40)),
  true_site_sigma = c(
    rep(0.5, 260), rep(NA, 5), rep(0, 5), rep(1.5, 5), rep(0, 5), rep(1.5, 5), rep(0.5, 15)
  ),
  true_batch_sigma = c(
    rep(0.3, 260), rep(NA, 5), rep(0, 5), rep(0, 5), rep(1.5, 5), rep(1.5, 5), rep(0.3, 15)
  ),
  stringsAsFactors = FALSE
)

write.csv(counts_c, file.path(OUTDIR, "design_c", "counts.csv"), row.names = TRUE)
write.csv(meta_c, file.path(OUTDIR, "design_c", "metadata.csv"), row.names = FALSE)
write.csv(manifest_c, file.path(OUTDIR, "design_c", "gene_manifest.csv"), row.names = FALSE)
cat("Design C written:", n_genes_c, "genes x", n_c, "samples\n")

cat("\nAll synthetic data generated in:", normalizePath(OUTDIR), "\n")