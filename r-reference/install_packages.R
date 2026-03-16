#!/usr/bin/env Rscript

# Set CRAN mirror (UK mirror 63)
options(repos = c(CRAN = "https://cran.ma.imperial.ac.uk/"))

# Run this AFTER creating the conda environment

cat("Installing dreamlet via BiocManager...\n\n")

# Install BiocManager if needed
if (!require("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cran.ma.imperial.ac.uk/")
}

cat("Installing dreamlet...\n")
BiocManager::install("dreamlet")

cat("\nInstalling zellkonverter (for H5AD support)...\n")
BiocManager::install("zellkonverter")

cat("\nInstalling rhdf5 (for H5AD support)...\n")
BiocManager::install('rhdf5')

cat("\nInstalling HDF5Array (for H5AD support)...\n")
BiocManager::install('HDF5Array')

# Verify installation
cat("\n=== Verification ===\n")
if (requireNamespace("dreamlet", quietly = TRUE)) {
    library(dreamlet)
    cat("✓ dreamlet version:", as.character(packageVersion("dreamlet")), "\n")
} else {
    cat("✗ dreamlet installation failed\n")
}

if (requireNamespace("zellkonverter", quietly = TRUE)) {
    cat("✓ zellkonverter version:", as.character(packageVersion("zellkonverter")), "\n")
} else {
    cat("✗ zellkonverter installation failed\n")
}

if (requireNamespace("rhdf5", quietly = TRUE)) {
    cat("✓ rhdf5 version:", as.character(packageVersion("rhdf5")), "\n")
} else {
    cat("✗ rhdf5 installation failed\n")
}

if (requireNamespace("HDF5Array", quietly = TRUE)) {
    cat("✓ HDF5Array version:", as.character(packageVersion("HDF5Array")), "\n")
} else {
    cat("✗ HDF5Array installation failed\n")
}

cat("\nSetup complete! You can now launch RStudio:\n")
cat("  open -a RStudio\n")