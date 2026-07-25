$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Remove-Item ".\STAGE1_DUAL_SOLVER_V20_RESULTS_bundle" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item ".\STAGE1_DUAL_SOLVER_V20_RESULTS_bundle.zip" -Force -ErrorAction SilentlyContinue

python tools/build_dual_solver_v20_bundle.py `
  --project-root . `
  --bundle-name STAGE1_DUAL_SOLVER_V20_RESULTS_bundle `
  --output STAGE1_DUAL_SOLVER_V20_RESULTS_bundle.zip

Get-Item .\STAGE1_DUAL_SOLVER_V20_RESULTS_bundle.zip | Select-Object FullName,Length
Get-FileHash .\STAGE1_DUAL_SOLVER_V20_RESULTS_bundle.zip -Algorithm SHA256
python -c "import zipfile; p='STAGE1_DUAL_SOLVER_V20_RESULTS_bundle.zip'; z=zipfile.ZipFile(p); bad=z.testzip(); print('FILE_COUNT=',len(z.namelist())); print('BAD_FILE=',bad); assert bad is None; print('ZIP_VALIDATION_PASS=True')"
