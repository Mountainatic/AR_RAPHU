"""Publish full development metrics averaged over five individual forward runs."""

import csv

from common import (CLASSICAL, CORE, ENTITIES, METRICS, NEURAL, PRISM, ROOT, SEEDS, matched, np,
                    read, registration, sha, write)

MODELS = (*CORE,*CLASSICAL,*NEURAL,*PRISM)
LABELS = {"SRU_H2S":"SRU H₂S", "CZ_Rod1":"CZ Rod1", "CZ_Rod2":"CZ Rod2",
          "Persistence":"持续值", "ElasticNet":"Elastic Net", "RBF_SVR":"RBF-SVR",
          "RandomForest":"随机森林", "HistGradientBoosting":"直方图梯度提升",
          "Transformer_original":"Transformer 原网格", "Transformer_expanded":"Transformer 扩展网格",
          "v0.6_KCW":"v0.6 KCW", "v0.6_KCWA":"原版 v0.6 KCWA", "KCW_fixed_A":"KCW + 固定 A",
          "Persistence_QC32":"持续值 + QC32", "KCW_QC32":"KCW + QC32"}


def label(value):
    return LABELS.get(value,value)


def expected_rows(entity):
    _,start,stop=matched.GEOMETRY[entity]
    return [start,stop]


def load(entity,model):
    if model in CLASSICAL:
        return read(ROOT/"classical"/entity/model/"RESULT.json")
    if model in NEURAL:
        return read(ROOT/"neural"/entity/model/"RESULT.json")
    return read(ROOT/"core"/entity/f"{model}.json")


def validate_completed(name,count):
    root=ROOT/name
    result=read(root/"COMPLETE.json")
    assert result["result_count"]==count
    assert result["freeze"]==sha(root/"ALL_FROZEN.json")
    freeze=read(root/"ALL_FROZEN.json")
    assert freeze["registration"]==sha(root/"REGISTRATION.json")
    assert all(sha(root/p/"FROZEN.json")==digest for p,digest in freeze["receipts"].items())


def summary(row):
    out={}
    for key in METRICS:
        values=np.array([r[key] for r in row["runs"]])
        assert np.isfinite(values).all()
        fixed=np.all(values==values[0])
        out[key]=dict(mean=float(values[0] if fixed else values.mean()),sd=float(0. if fixed else values.std(ddof=1)))
    return out


def write_csv(path,fields,rows):
    tmp=path.with_suffix(".tmp")
    with tmp.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)
    tmp.replace(path)


def cell(metric,item):
    spec=".3e" if metric=="MSE" else ".6f"
    mean=format(item["mean"],spec)
    if item["sd"]==0.:
        return mean
    return f"{mean} ± {item['sd']:.2g}"


def report(per_entity, nonconverged, gram):
    lines=["# PRISM 同协议五次运行完整基线", "",
           r"SRU H₂S、CZ Rod1 和 CZ Rod2 使用 \(H=0,D=4,L=128\)。过程量可见至 \(t\)，真值可见至 \(t-4\)。每套数据使用同一组开发评分行。",
           "",
           "逐次在原目标尺度计算 RMSE、MAE、MSE 和水平 R²，然后对五次得分取算术均值。表格中的 ± 为五次得分的样本标准差。只显示均值的单元格标准差为零。",
           "",
           "- 新增 PCR、PLS、Elastic Net、RBF-SVR、随机森林、直方图梯度提升。统一使用 128 步过程量与成熟目标历史。在两组带时间隔离的向前训练折上选择超参数和水平或增量目标，再各自完整重训五次。两个树模型使用种子 0–4，其他四个方法按确定性算法重训。",
           "- AR 和 ARX 沿用训练折选出的规格，各完整重训五次。神经基线沿用原训练折选出的配置和训练轮数，重放种子 0、1、2，并重新训练种子 3、4。各次独立评分。",
           "- 持续值和 PRISM 各版本使用五次确定性推理回放。标准差刻画这些复跑的变化。", ""]
    if nonconverged:
        lines.extend([f"经典基线有 {nonconverged} 次候选折拟合达到迭代上限，相关候选退出选型。最终配置的两个训练折和五次复跑均完成。", ""])
    if gram:
        lines.extend(["两根晶棒的 Elastic Net 使用预计算 Gram 矩阵。目标函数、候选网格和收敛容差一致，SRU 训练段上的两条求解路径预测最大绝对差为 "
                      f"{gram['max_abs']:.2e}。", ""])
    for entity,rows in per_entity.items():
        ref=next(row for row in rows if row["model"]=="v0.6_KCWA")
        best_new=min((row for row in rows if row["model"] in CLASSICAL),key=lambda r:r["summary"]["RMSE"]["mean"])
        best=min((row for row in rows if row["model"] not in PRISM),key=lambda r:r["summary"]["RMSE"]["mean"])
        lines.extend([f"## {label(entity)}，{ref['N']:,} 个开发评分点", "",
                      f"新增经典方法中 {label(best_new['model'])} 的平均 RMSE 最低，为 {best_new['summary']['RMSE']['mean']:.6f}。全部外部基线中 {label(best['model'])} 最低，为 {best['summary']['RMSE']['mean']:.6f}。", "",
                      "| 模型 | RMSE | MAE | MSE | R² |", "| --- | ---: | ---: | ---: | ---: |"])
        for row in sorted(rows,key=lambda r:r["summary"]["RMSE"]["mean"]):
            name=label(row["model"])
            if row is best:
                name="**"+name+"**"
            values=[cell(key,row["summary"][key]) for key in METRICS]
            lines.append("| "+" | ".join([name,*values])+" |")
        lines.append("")
    lines.extend(["## 结果文件", "", "`MEAN_METRICS.csv` 保存五次运行的均值和标准差，`RUN_METRICS.csv` 保存每一个种子的四项得分，`RESULTS.json` 保存配置和验证元数据。", "",
                  "本阶段的工作目录保留各次拟合的校验记录。"])
    (ROOT/"REPORT_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run():
    reg=registration()
    validate_completed("classical",len(ENTITIES)*len(CLASSICAL))
    validate_completed("neural",len(ENTITIES)*len(NEURAL))
    full={}
    nonconverged=0
    gram=None
    for entity in ENTITIES:
        core=ROOT/"core"/entity
        stop=read(core/"COMPLETE.json")
        assert stop==dict(records=len(CORE)+len(PRISM),freeze=sha(core/"FROZEN.json"))
        full[entity]=[]
        for model in MODELS:
            row=load(entity,model)
            if model in CLASSICAL:
                folder=ROOT/"classical"/entity/model
                ledger=read(folder/"SELECTION.json")
                assert row["selected"]==ledger["selected"]
                nonconverged+=sum(fold["status"]=="DID_NOT_CONVERGE" for c in ledger["candidates"] for fold in c["folds"])
                amendment=read(folder/"FROZEN.json").get("solver_amendment_sha256")
                if amendment:
                    proof=ROOT/"elastic_gram"/"EQUIVALENCE.json"
                    assert sha(proof)==amendment
                    gram=read(proof)
            assert row["entity"]==entity and row["model"]==model
            assert row["development"]==expected_rows(entity) and row["N"]==np.diff(row["development"])[0]
            assert tuple(v["seed"] for v in row["runs"])==SEEDS
            for v in row["runs"]:
                np.testing.assert_allclose(v["RMSE"]**2,v["MSE"],rtol=1e-14,atol=0)
            full[entity].append(dict(**row,summary=summary(row)))
    models=[value for group in full.values() for value in group]
    write_csv(ROOT/"MEAN_METRICS.csv",["entity","N","model","kind","runs",*(k+"_"+stat for k in METRICS for stat in ("mean","sd"))],
              ((r["entity"],r["N"],r["model"],r["kind"],len(r["runs"]),*(r["summary"][k][s] for k in METRICS for s in ("mean","sd"))) for r in models))
    write_csv(ROOT/"RUN_METRICS.csv",["entity","N","model","kind","seed",*METRICS],
              ((r["entity"],r["N"],r["model"],r["kind"],s["seed"],*(s[k] for k in METRICS)) for r in models for s in r["runs"]))
    write(ROOT/"RESULTS.json",dict(status="COMPLETED",registration=reg,nonconverged_candidate_folds=nonconverged,
                                  ElasticNet_Gram=gram,per_entity=full))
    report(full,nonconverged,gram)
    print("COMPLETE",len(models),"model–dataset combinations",len(SEEDS)*len(models),"runs",flush=True)


if __name__=="__main__":
    run()
