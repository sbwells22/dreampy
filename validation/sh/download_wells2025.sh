#!/usr/bin/env bash
# download_wells2025.sh — Download the Wells et al. 2025 T cell dataset.
#
# Reference:
#   Wells SB, Rainbow DB, Mark M, Szabo PA, et al. "Multimodal profiling
#   reveals tissue-directed signatures of human immune cells altered with age."
#   Nature Immunology 26, 1612-1625 (2025).
#
# Source: CellxGene Census
#   Dataset ID: 965008a7-e698-413c-a746-8855a945a7c5
#   T cell subset, ~24 donors, 13 T cell types, donor x tissue pseudobulk
#
# Usage:
#   bash download_wells2025.sh [output_directory]
#
# Default output: ./data/wells2025/

set -euo pipefail

URL="https://datasets.cellxgene.cziscience.com/965008a7-e698-413c-a746-8855a945a7c5.h5ad"
FILENAME="wells2025.h5ad"
# TODO: populate after first verified download
SHA256="de47634244ff060af7273e4485816326734b36dbabe78253b8c2e70bbf0181bc"

OUTDIR="${1:-../data/wells2025}"
OUTPATH="${OUTDIR}/${FILENAME}"

mkdir -p "${OUTDIR}"

if [ -f "${OUTPATH}" ]; then
    echo "File already exists: ${OUTPATH}"
    if [ -n "${SHA256}" ]; then
        echo "Verifying checksum..."
        echo "${SHA256}  ${OUTPATH}" | shasum -a 256 --check --status && {
            echo "Checksum OK. Skipping download."
            exit 0
        } || {
            echo "Checksum mismatch. Re-downloading."
        }
    else
        echo "No checksum available. Skipping download. Delete file to force re-download."
        exit 0
    fi
fi

echo "Downloading Wells et al. 2025 T cell dataset..."
echo "Source ${URL}"
echo "Destination ${OUTPATH}"
echo ""
echo "NOTE: This file is ~5 GB. Download depends on connection speed."
echo ""

curl -L --progress-bar -o "${OUTPATH}" "${URL}"

if [ -n "${SHA256}" ]; then
    echo ""
    echo "Verifying checksum..."
    echo "${SHA256}  ${OUTPATH}" | shasum -a 256 --check || {
        echo "ERROR: checksum verification failed."
        echo "The downloaded file may be corrupted. Delete and retry."
        exit 1
    }
else
    echo ""
    echo "WARNING: No checksum defined. Compute and add to this script:"
    echo "  shasum -a 256 ${OUTPATH}"
fi

echo ""
echo "Download complete: ${OUTPATH}"
