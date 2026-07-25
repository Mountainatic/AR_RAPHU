param(
  [string]$Devices = "0",
  [int]$WorkersPerGpu = 4,
  [switch]$SkipProfiler
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Out = ".\results_stage1\STAGE1_DUAL_SOLVER_V20"

Write-Host "=== Clean old V20 outputs ==="
Remove-Item $Out -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $Out | Out-Null

Write-Host "=== Environment ==="
python -c "import torch,platform,json; print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},indent=2))" | Tee-Object "$Out\torch_env.json"
nvidia-smi | Tee-Object "$Out\nvidia_smi_before.txt"

Write-Host "=== Full pytest ==="
$pytestLog = "$Out\pytest_full_output.txt"
python -m pytest tests/test_stage1.py tests/test_stage1_acceleration.py tests/test_stage1_dual_solver_v20.py -q 2>&1 | Tee-Object $pytestLog
$pytestExit = $LASTEXITCODE
python tools/write_pytest_summary.py --input $pytestLog --output "$Out\pytest_summary.json" --exit-code $pytestExit
if ($pytestExit -ne 0) { throw "pytest failed" }

if (-not $SkipProfiler) {
  Write-Host "=== CUDA profiler ==="
  python tools/profile_v20_gpu.py --steps 50 --output results_stage1/STAGE1_DUAL_SOLVER_V20/gpu_profile
}

Write-Host "=== KAN warmup jobs ==="
python run_kan_fast_s0_v20.py --mode manifest --device cuda
python tools/run_gpu_job_pool.py `
  --manifest "$Out\kan_fast\manifests\warmup_jobs.json" `
  --devices $Devices --workers-per-device $WorkersPerGpu --resume

Write-Host "=== KAN independent pruning forks ==="
python tools/run_gpu_job_pool.py `
  --manifest "$Out\kan_fast\manifests\fork_jobs.json" `
  --devices $Devices --workers-per-device $WorkersPerGpu --resume
python run_kan_fast_s0_v20.py --mode aggregate --device cuda

Write-Host "=== Variational seed-0 screen ==="
python run_variational_stage1_s0_v20.py --mode manifest --device cuda
python tools/run_gpu_job_pool.py `
  --manifest "$Out\variational\manifests\variational_screen_jobs.json" `
  --devices $Devices --workers-per-device $WorkersPerGpu --resume

Write-Host "=== Variational five-seed formal candidates ==="
python run_variational_stage1_s0_v20.py --mode select-screen --device cuda
python tools/run_gpu_job_pool.py `
  --manifest "$Out\variational\manifests\variational_formal_jobs.json" `
  --devices $Devices --workers-per-device $WorkersPerGpu --resume
python run_variational_stage1_s0_v20.py --mode aggregate --target clean --device cuda

Write-Host "=== Variational noisy selected configuration ==="
python run_variational_stage1_s0_v20.py --mode make-noisy --device cuda
python tools/run_gpu_job_pool.py `
  --manifest "$Out\variational\manifests\variational_noisy_jobs.json" `
  --devices $Devices --workers-per-device $WorkersPerGpu --resume
python run_variational_stage1_s0_v20.py --mode aggregate --target observed --device cuda

Write-Host "=== Final comparison and audit ==="
python run_stage1_dual_solver_v20.py --phase finalize
nvidia-smi | Tee-Object "$Out\nvidia_smi_after.txt"

Write-Host "=== Package final results ==="
& ".\PACKAGE_V20_RESULTS.ps1"
