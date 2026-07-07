#!/usr/bin/env bash
set -euo pipefail

pip install -r requirements.txt

rm -rf /tmp/ttdata
git clone --depth 1 "https://x-access-token:${DATA_REPO_TOKEN}@github.com/ads40788/turkey-trot-research.git" /tmp/ttdata

mkdir -p db
cp /tmp/ttdata/db/turkey_trot.db db/turkey_trot.db
rm -rf /tmp/ttdata
