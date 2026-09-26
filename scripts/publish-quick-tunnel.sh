#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
URL="${1:-}"
VERCEL_CLI="/home/paul/.nvm/versions/node/v20.19.5/lib/node_modules/vercel/dist/vc.js"

if [[ ! "$URL" =~ ^https://[a-z0-9-]+\.trycloudflare\.com/?$ ]]; then
    printf 'Usage: %s https://your-tunnel.trycloudflare.com\n' "$0" >&2
    exit 2
fi

cd "$ROOT"
if [[ -n "$(git status --porcelain -- public/backend.json)" ]]; then
    printf 'public/backend.json has uncommitted changes; refusing to overwrite it.\n' >&2
    exit 1
fi

URL="${URL%/}"
printf '{"url":"%s"}\n' "$URL" > public/backend.json
git add -- public/backend.json

if ! git diff --cached --quiet -- public/backend.json; then
    git commit -m "Point frontend to current local backend" -- public/backend.json
    git push origin main
fi

VERCEL_TELEMETRY_DISABLED=1 /usr/bin/node "$VERCEL_CLI" deploy --prod --yes
printf 'Published local backend %s to the Vercel frontend.\n' "$URL"
