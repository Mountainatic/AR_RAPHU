# Codex 执行提示词：Spectral PS-AR-RAPHU v0.3.1 表示修复与核心重启

你现在要在现有 Spectral PS-AR-RAPHU v0.3 代码上实施 **v0.3.1 表示修复**。不要从零重写项目，不要覆盖旧结果，不要自行推断任何 basis、阈值、目标或下一阶段。

必须先阅读：

```text
Spectral_PS_AR_RAPHU_Theory_v0_3_1.md
Spectral_PS_AR_RAPHU_Validation_Plan_v0_3_1.md
```

## 一、当前冻结事实

```text
E0_COMPONENT_IDENTITY_PASS
maximum identity error = 1.7763568394002505e-15

E1_PROJECTION_FALLBACK_FAIL
12x16 worst NRMSE:
AR-S1 0.1470905246
AR-S2 0.4474544259
AR-S3 0.4649135705
AR-S4 0.4372020404

E2–E8 NOT RUN
```

旧结果必须保留：

```text
results/spectral_v03/
configs/spectral_v03.yaml
```

不得修改、删除或伪装重跑。

## 二、本轮只做

```text
1. 新增 v0.3.1 config 和 protocol revision
2. 新增高效 tensor surface projection
3. 实现并运行 E1R
4. E1R 通过后实现并运行 E2A
5. E2A 通过后运行 E2B
6. E2B 通过后实现并运行 E3
7. 生成 V031_CORE_DECISION.md
8. 暂停并打包
```

禁止运行 E4–E8。

## 三、文件

新增：

```text
PROTOCOL_REVISION_V031.md
configs/spectral_v031.yaml
src/ar_raphu/spectral/projection.py
tests/test_spectral_projection_repair.py
results/spectral_v031/
```

扩展：

```text
tools/run_spectral_suite.py
tools/summarize_spectral_suite.py
src/ar_raphu/spectral/contracts.py
src/ar_raphu/spectral/solver.py
src/ar_raphu/spectral/gram_svd.py
```

不得恢复：

```text
training.prune_external_path
group-prox support selection
A-support-only B/M8 dispatch
```

## 四、E1R

固定：

```text
M_x = 16
lag cubic B-spline candidates = 24,28,32,40
identity lag reference = np.eye(64)
scenarios = AR-S1,AR-S2,AR-S3,AR-S4
seeds = 0,1,2,3,4
variables = truth active support only
```

投影使用两侧最小二乘：

```python
intermediate = np.linalg.lstsq(lag_basis, centered_truth, rcond=None)[0]
theta_t = np.linalg.lstsq(amplitude_basis, intermediate.T, rcond=None)[0]
theta = theta_t.T
estimate = lag_basis @ theta @ amplitude_basis.T
```

不得构建巨大 Kronecker design。

结构级通过：

```text
worst NRMSE <= 0.05
AND every scenario error <= 2 * identity-reference scenario error
```

必须选中：

```text
32 x 16
```

回归预期和容差全部从 `spectral_v031.yaml` 读取，不得写死另一套值。

E1R 不通过立即停止。

## 五、E2A

每个 active variable 单独运行。

目标：

```python
target = components.x_contribution_by_variable[:, variable]
```

模型只含该变量的 \(32\times16\) full kernel。

禁止：

```text
完整 y 作为目标
AR branch
其他变量
support penalty
truth kernel coefficients as initialization
```

truth 只提供 target component 和最终评价 surface。

smoothing 只按 validation contribution MSE 选择。

维数 512，CPU FP64 Cholesky。

通过规则严格按验证方案。

## 六、E2B

目标：

```python
target = components.x_total_contribution
```

模型只含 oracle active support 三个变量。

不含 AR。

维数 1536，CPU FP64 Cholesky。

通过规则严格按验证方案。

## 七、E3

全部十个变量进入。

比较：

```text
O oracle AR subtraction
Y only-y residualization
D double residualization
J joint convex AR+X
```

D 必须同时残差化：

```python
y_residual = y - mu_hat
phi_residual = phi - pi_hat
```

前向 cross-fit：

```text
initial prefix = 2000 targets
folds = 4
purge = 65 targets
selection tail = 20% of nuisance train prefix
never use future fold
```

主 basis 固定 \(32\times16\)。不得重新选择 basis。

十变量主求解用 FP64 PCG：

```text
relative residual <= 1e-8
max iterations = 2000
block-Jacobi preconditioner
```

PCG 不收敛的配置不得用于结论。

## 八、测试

只运行验证方案指定的九个测试文件。

必须新增：

```text
identity basis test
two-sided projection vs explicit Kronecker test
E1R regression test
E2A target semantics test
E2B target semantics test
double residual target/design test
PCG vs direct solver test
```

不要运行全仓库测试、旧 M7/M8 审计、SHA、manifest 或 HTML。

## 九、运行

```bash
python -m pytest \
  tests/test_ar_raphu_model.py \
  tests/test_sequence_training.py \
  tests/test_spectral_contracts.py \
  tests/test_synthetic_components.py \
  tests/test_spectral_design.py \
  tests/test_spectral_projection_repair.py \
  tests/test_double_residualization.py \
  tests/test_spectral_solver.py \
  tests/test_gram_svd.py \
  -q

python tools/run_spectral_suite.py \
  --config configs/spectral_v031.yaml \
  --experiment E1R \
  --stage development \
  --device cpu
```

只有 E1R 通过才运行：

```bash
python tools/run_spectral_suite.py \
  --config configs/spectral_v031.yaml \
  --experiment E2A \
  --stage development \
  --device cpu

python tools/run_spectral_suite.py \
  --config configs/spectral_v031.yaml \
  --experiment E2B \
  --stage development \
  --device cpu
```

只有 E2A/E2B 通过才运行：

```bash
python tools/run_spectral_suite.py \
  --config configs/spectral_v031.yaml \
  --experiment E3 \
  --stage development \
  --device cuda
```

## 十、结果和停止

生成：

```text
results/spectral_v031/E1R/
results/spectral_v031/E2A/
results/spectral_v031/E2B/
results/spectral_v031/E3/
results/spectral_v031/V031_CORE_DECISION.md
results/spectral_v031/spectral_v031_core_summary.csv
```

`V031_CORE_DECISION.md` 只能按预注册映射生成。

完成 E3 后停止，不实现 E4–E8。

## 十一、打包

删除旧的同名输出 zip，然后：

```bash
rm -f SPECTRAL_PS_AR_RAPHU_V031_CORE_RESULTS.zip

zip -r SPECTRAL_PS_AR_RAPHU_V031_CORE_RESULTS.zip \
  PROTOCOL_REVISION_V031.md \
  configs/spectral_v031.yaml \
  src/ar_raphu/spectral \
  tools/run_spectral_job.py \
  tools/run_spectral_suite.py \
  tools/summarize_spectral_suite.py \
  tests/test_spectral_*.py \
  results/spectral_v031

test -f results/spectral_v031/V031_CORE_DECISION.md
test -f results/spectral_v031/spectral_v031_core_summary.csv
unzip -t SPECTRAL_PS_AR_RAPHU_V031_CORE_RESULTS.zip
```

不生成 SHA 或 manifest。

最终只返还：

```text
SPECTRAL_PS_AR_RAPHU_V031_CORE_RESULTS.zip
```

并在终端打印：

```text
FINAL_PACKAGE=<absolute path>
NEXT_ALLOWED_STAGE=<value from V031_CORE_DECISION.md>
```
