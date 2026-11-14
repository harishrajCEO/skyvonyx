#!/usr/bin/env bash
# Shallow-clone the repos the Skyvonyx stack is built on into vendor/.
# Idempotent: existing clones are left alone. Run from the repo root.
#
#   harishrajCEO/sahi                 your SAHI fork (slicing + GreedyNMM re-stitch)
#   harishrajCEO/rf-detr              your RF-DETR fork (permissive-licence detector, optional)
#   harishrajCEO/gods-eye-view        fork of bilawalsidhu/gods-eye-view: Cesium globe UI that consumes GeoJSON
#   harishrajCEO/ECC                  agent skills / automation collection
#   facebookresearch/sam2             SAM 2 (the segmenter actually used)
#   facebookresearch/segment-anything SAM 1 (fallback segmenter)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor
for repo in harishrajCEO/sahi harishrajCEO/rf-detr harishrajCEO/gods-eye-view harishrajCEO/ECC \
            facebookresearch/sam2 facebookresearch/segment-anything; do
  name=$(basename "$repo")
  if [ -d "vendor/$name/.git" ]; then
    echo "vendor/$name already present"
  else
    echo "cloning $repo"
    git clone --depth 1 -q "https://github.com/$repo.git" "vendor/$name"
  fi
done
echo "done. vendor/ is gitignored; re-run this script on a fresh checkout."
