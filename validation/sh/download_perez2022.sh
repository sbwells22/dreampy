#!/usr/bin/env bash
# download_perez2022.sh — Download the Perez et al. 2022 SLE dataset.
#
# Reference:
#   Perez RK, Gordon MG, Suber M, et al. "Single-cell RNA-seq reveals cell
#   type-specific molecular and genetic associations to lupus."
#   Science 376, eabf1970 (2022).
#
# Source: CellxGene Census
#   Dataset ID: 4118e166-34f5-4c1f-9eed-c64b90a3dace
#   1.26M cells, 261 donors (162 SLE, 99 control), 11 cell types
#
# Usage:
#   bash download_perez2022.sh [output_directory]
#
# Default output: ./data/perez2022/

set -euo pipefail

URL="https://datasets.cellxgene.cziscience.com/4118e166-34f5-4c1f-9eed-c64b90a3dace.h5ad"
FILENAME="perez2022.h5ad"
# TODO: populate after first verified download
SHA256="7ba85edbdc033a9aeeecabe7a1cfcb0660282cc24b04f49fb52dd3825052c620"

OUTDIR="${1:-../data/perez2022}"
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

echo "Downloading Perez et al. 2022 SLE dataset..."
echo "Source ${URL}"
echo "Destination ${OUTPATH}"
echo ""
echo "NOTE: This file is ~11 GB. Download depends on connection speed."
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
