#!/usr/bin/env bash
# upload_raw.sh — copy Dominick's CSVs to GCS with normalized names.
# Usage: BUCKET=my-bucket ./upload_raw.sh   (run from the directory holding the CSVs)
set -euo pipefail

: "${BUCKET:?Set BUCKET, e.g. BUCKET=my-bucket ./upload_raw.sh}"

# Keep the Mac awake for the duration of this script (replaces `caffeinate -i bash -c`)
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i -w $$ &
fi

shopt -s nullglob

upload() {
  local prefix="$1" dest="$2" f n
  for f in "${prefix}"_*.csv; do
    # strip prefix, lowercase, spaces -> _, dashes -> _
    n=$(echo "${f#"${prefix}"_}" | tr 'A-Z ' 'a-z_' | sed 's/-/_/g')
    gcloud storage cp --no-clobber "$f" "gs://${BUCKET}/raw/${dest}/${n}"
  done
}

upload data movement
upload desc upc_desc