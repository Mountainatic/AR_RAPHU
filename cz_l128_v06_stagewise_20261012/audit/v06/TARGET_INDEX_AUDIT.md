# Phase 2 — 真实目标索引

权威 _sample_rows：预测原点 t 是历史排他末尾。历史 [t-128,t)，锚点 [t-1,t)，未来窗口 [t+h-1,t+h)。项目 H1 为 D[t]-D[t-1]；H4 为 D[t+3]-D[t-1]，W=W0=1。与母版窗口等价时母版 h = 项目 h-1；H1 对应母版 h=0。项目旧 h=0 别名映射 effective_h=1。L128 正式运行注册 H1；H4 只读审计没有新增 H4 训练。

手工例：t=5，记录 D[i]=i²/100，最后可见 index=4。H1 anchor=4,target=5,value=.25-.16=.09,maturity=5；H4 anchor=4,target=8,value=.64-.16=.48,maturity=8。2 秒约定下 target_ready/feedback=10 秒或16 秒；这是协议坐标，不是记录到的仪器交付时间。默认 delivery_delay=0。

CSV 包含逐行 entity/segment/split/head、origin、anchor/target 窗口、input/output 可见末尾、label_maturity、timestamp 证据、ready/feedback、horizon/stride/formula、raw history 依赖及 legal_support/invalid_reason。默认 origin 位于当前实体，禁止目标作为 predictor。对短段、跨时间缺口、不同 split 依赖、purge 标志、棒身份、末尾未知未来控制均只报告。跨 split 警告需要原协议审查，不自动认定合法的过去历史跨边界为标签泄漏。

真实逐行产物在 artifacts/real/<direction>/H1_{train,validation}_target_alignment.csv 与 H4 同名文件。H1 比较已有 y_true；H4 在既有 H1 origins 按旧 H4 公式生成只读对照，尾部不足标非法，不改 H1 行。A 成熟规则由原 a_state.cz_target_end 与 feature_rows 执行；本次不替代。

