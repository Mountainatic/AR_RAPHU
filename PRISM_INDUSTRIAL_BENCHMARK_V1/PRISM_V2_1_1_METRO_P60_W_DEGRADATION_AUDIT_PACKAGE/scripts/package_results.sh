#!/usr/bin/env bash
set -euo pipefail
: "${RESULT_ROOT:?Set RESULT_ROOT}"
: "${CODE_ROOT:?Set CODE_ROOT}"
: "${DESIGN_ROOT:?Set DESIGN_ROOT}"
OUT_PARENT="${OUT_PARENT:-$(pwd)}"
BUNDLE_NAME="PRISM_V2_1_1_METRO_P60_W_DEGRADATION_AUDIT_RESULTS_bundle"
STAGING="$OUT_PARENT/$BUNDLE_NAME"
ZIP_PATH="$OUT_PARENT/$BUNDLE_NAME.zip"
SHA_PATH="$ZIP_PATH.sha256"
rm -rf "$STAGING" "$ZIP_PATH" "$SHA_PATH"
mkdir -p "$STAGING/design" "$STAGING/results" "$STAGING/code_audit"
cp -a "$DESIGN_ROOT"/. "$STAGING/design/"
cp -a "$RESULT_ROOT"/. "$STAGING/results/"
(
 cd "$CODE_ROOT"
 git rev-parse HEAD > "$STAGING/code_audit/GIT_HEAD.txt"
 git branch --show-current > "$STAGING/code_audit/GIT_BRANCH.txt"
 git status --porcelain=v1 > "$STAGING/code_audit/GIT_STATUS_PORCELAIN.txt"
 git diff --binary > "$STAGING/code_audit/WORKTREE.diff"
 git diff --cached --binary > "$STAGING/code_audit/INDEX.diff"
 git log -n 20 --oneline --decorate > "$STAGING/code_audit/GIT_LOG_LAST20.txt"
)
python - "$STAGING" <<'PYMANIFEST'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]); files=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and p.name not in {'MANIFEST.json','SHA256SUMS.txt'}:
  b=p.read_bytes(); files.append({'path':p.relative_to(root).as_posix(),'size':len(b),'sha256':hashlib.sha256(b).hexdigest()})
(root/'MANIFEST.json').write_text(json.dumps({'file_count':len(files),'files':files},indent=2),encoding='utf-8')
with (root/'SHA256SUMS.txt').open('w',encoding='utf-8') as f:
 for x in files: f.write(f"{x['sha256']}  {x['path']}\n")
PYMANIFEST
(cd "$OUT_PARENT" && zip -q -r "$ZIP_PATH" "$BUNDLE_NAME")
unzip -t "$ZIP_PATH"
sha256sum "$ZIP_PATH" > "$SHA_PATH"
printf 'ZIP=%s\n' "$(readlink -f "$ZIP_PATH")"
printf 'SIZE_BYTES=%s\n' "$(stat -c %s "$ZIP_PATH")"
printf 'SHA256=%s\n' "$(cat "$SHA_PATH")"
