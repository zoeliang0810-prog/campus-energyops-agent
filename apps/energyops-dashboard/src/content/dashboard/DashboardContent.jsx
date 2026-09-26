import React, { useEffect, useMemo, useState } from "react";

import {
  Button, DataComponent, DataTable, EvidenceChart, MetricCard, Section,
  useDataApp, useDashboardTabs,
} from "../../data-app-public.jsx";
import "./example.css";

const powerPlanSpec = {
  type: "line", x: "hour", y: "plannedLoadKw",
  fields: ["plannedLoadKw", "plannedPvKw", "plannedGridImportKw"],
  colors: { plannedLoadKw: "var(--chart-1)", plannedPvKw: "var(--positive)", plannedGridImportKw: "var(--chart-5)" },
  legend: { labels: { plannedLoadKw: "负荷计划", plannedPvKw: "光伏计划", plannedGridImportKw: "电网购电计划" } },
  valueDecimals: 0, showLegend: true, showXAxisLabel: false, showYAxisLabel: false, stackable: false,
};

const storagePlanSpec = {
  type: "line", x: "hour", y: "plannedBatteryChargeKw",
  fields: ["plannedBatteryChargeKw", "plannedBatteryDischargeKw", "plannedBatteryEnergyKwh"],
  colors: { plannedBatteryChargeKw: "var(--chart-2)", plannedBatteryDischargeKw: "var(--chart-4)", plannedBatteryEnergyKwh: "var(--chart-3)" },
  legend: { labels: { plannedBatteryChargeKw: "充电功率", plannedBatteryDischargeKw: "放电功率", plannedBatteryEnergyKwh: "储能电量" } },
  valueDecimals: 0, showLegend: true, showXAxisLabel: false, showYAxisLabel: false, stackable: false,
};

const deviationSpec = {
  type: "line", x: "hour", y: "plannedGridImportKw",
  fields: ["plannedGridImportKw", "actualGridImportKw"],
  colors: { plannedGridImportKw: "var(--chart-1)", actualGridImportKw: "var(--negative)" },
  legend: { labels: { plannedGridImportKw: "计划购电", actualGridImportKw: "实际购电" } },
  valueDecimals: 0, showLegend: true, showXAxisLabel: false, showYAxisLabel: false, stackable: false,
};

function numberCell(value) {
  return Number.isFinite(value) ? new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 1 }).format(value) : "—";
}
function nullableNumberCell(value) {
  return Number.isFinite(value) ? numberCell(value) : <span className="missing-value">未接入</span>;
}
function percentCell(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : <span className="missing-value">无法计算</span>;
}
function nullableTextCell(value) {
  return value ? String(value) : <span className="missing-value">无</span>;
}
function statusCell(value) {
  const normalized = String(value ?? "unknown").toLowerCase();
  const tone = normalized.includes("pass") || normalized.includes("complete") || normalized.includes("success")
    ? "good" : normalized.includes("warn") || normalized.includes("pending") ? "warning" : "neutral";
  const labels = { pass: "通过", warning: "警告", completed: "完成", success: "成功", pending_human_approval: "待人工审批" };
  return <span className={`status-text ${tone}`}>{labels[normalized] ?? String(value ?? "未知")}</span>;
}
function cny(value) {
  return Number.isFinite(value) ? new Intl.NumberFormat("zh-CN", { style: "currency", currency: "CNY", maximumFractionDigits: 0 }).format(value) : "—";
}
function percent(value) { return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—"; }

const planColumns = [
  { field: "hour", label: "时段" },
  { field: "plannedLoadKw", label: "负荷 kW", renderCell: numberCell },
  { field: "plannedPvKw", label: "光伏 kW", renderCell: numberCell },
  { field: "plannedGridImportKw", label: "购电 kW", renderCell: numberCell },
  { field: "plannedBatteryChargeKw", label: "充电 kW", renderCell: numberCell },
  { field: "plannedBatteryDischargeKw", label: "放电 kW", renderCell: numberCell },
  { field: "plannedBatteryEnergyKwh", label: "储能电量 kWh", renderCell: numberCell },
];
const monitoringColumns = [
  { field: "hour", label: "时段" },
  { field: "plannedLoadKw", label: "计划负荷 kW", renderCell: numberCell },
  { field: "actualLoadKw", label: "实际负荷 kW", renderCell: nullableNumberCell },
  { field: "loadDeviationPct", label: "负荷偏差", renderCell: percentCell },
  { field: "actualPvKw", label: "实际光伏 kW", renderCell: nullableNumberCell },
  { field: "actualBatteryEnergyKwh", label: "实际储能电量 kWh", renderCell: nullableNumberCell },
  { field: "monitoringStatus", label: "监测状态", renderCell: () => <span className="status-text waiting">等待实际数据</span> },
];
const ruleColumns = [
  { field: "signal", label: "监测对象" }, { field: "threshold", label: "触发阈值" },
  { field: "persistence", label: "持续条件" }, { field: "action", label: "建议动作" },
  { field: "status", label: "状态", renderCell: () => <span className="status-text draft">草案</span> },
];
const qualityColumns = [
  { field: "dataset", label: "数据集" }, { field: "status", label: "状态", renderCell: statusCell },
  { field: "finding", label: "检查结果" }, { field: "evidenceId", label: "证据编号" },
];
const traceColumns = [
  { field: "sequence", label: "序号" }, { field: "eventType", label: "事件" },
  { field: "component", label: "组件" }, { field: "status", label: "状态", renderCell: statusCell },
  { field: "evidenceId", label: "证据编号", renderCell: nullableTextCell },
];
const toolColumns = [
  { field: "tool", label: "白名单工具" }, { field: "purpose", label: "作用" }, { field: "access", label: "权限" },
];
const presets = ["今天的调度是否可以执行？", "计划和实际偏差多大？", "什么时候应该重新调度？", "为什么当前计划还不能下发？"];

function interpretRequest(text, summary, scheduleRows, actualCount) {
  const request = text.trim();
  if (!request) return null;
  if (/执行|下发|控制|可用/.test(request)) return {
    intent: "执行安全检查",
    answer: "当前计划不能直接下发设备。它已通过数学与证据校验，但仍是仿真计划，且等待人工审批。",
    next: "先核验储能与光伏铭牌、电价和设备接口，再由授权人员审批；本界面不会发送控制指令。",
  };
  if (/偏差|实际|跟踪|监测/.test(request)) return {
    intent: "计划—实际偏差监测",
    answer: `当前收到 ${actualCount}/${scheduleRows.length} 个时段的实际数据，因此还不能计算可靠偏差。缺失值保留为“未接入”。`,
    next: "接入实际负荷、实际光伏和储能 SOC 后，按时段对齐并连续监测阈值。",
  };
  if (/重调度|重新调度|调整|异常/.test(request)) return {
    intent: "重调度建议",
    answer: "重调度规则骨架已经建立，但实际遥测尚未接入，当前不会生成误导性的触发结论。",
    next: "实际数据满足任一草案阈值并持续指定时段后，先生成建议，再走数据质量检查、优化、Verifier、Evidence 和人工审批。",
  };
  if (/费用|成本|省钱|经济/.test(request)) return {
    intent: "经济调度解释",
    answer: `当前仿真目标为经济性，单日参考费用为 ${cny(summary.totalCostCny)}，峰值购电约 ${numberCell(summary.peakImportKw)} kW。`,
    next: "该费用使用未完成现场核验的参考电价，只适合方案比较，不应作为结算依据。",
  };
  if (/光伏|储能|电池/.test(request)) return {
    intent: "光储运行解释",
    answer: `本计划的光伏自用率为 ${percent(summary.pvSelfConsumptionRatio)}，储能日吞吐量约 ${numberCell(summary.batteryThroughputKwh)} kWh。`,
    next: "继续在“调度计划”中查看逐时光伏、充放电功率和储能电量。",
  };
  return {
    intent: "一般能源问题", answer: "这个本地界面只做安全的只读解释，没有调用 DeepSeek API，也没有执行设备控制。",
    next: "可询问计划是否可执行、偏差大小、重调度条件、成本或光储运行。后续接入本地 Agent 服务后，再由个人 DeepSeek API 选择五个白名单工具。",
  };
}

function StatusStrip({ summary }) {
  const items = [
    ["运行模式", summary.simulationOnly ? "仿真" : "生产", "warning"],
    ["Verifier", summary.verificationPassed ? "已通过" : "未通过", summary.verificationPassed ? "good" : "bad"],
    ["执行权限", summary.executable ? "可执行" : "不可执行", summary.executable ? "good" : "bad"],
    ["审批状态", summary.releaseStatus === "pending_human_approval" ? "待人工审批" : summary.releaseStatus, "warning"],
  ];
  return <section className="status-strip" aria-label="计划安全状态">
    {items.map(([label, value, tone]) => <div key={label} className="status-strip-item"><span>{label}</span><strong data-tone={tone}>{value}</strong></div>)}
  </section>;
}

function NaturalLanguageConsole({ summary, scheduleRows, actualCount }) {
  const [request, setRequest] = useState(presets[0]);
  const [result, setResult] = useState(() => interpretRequest(presets[0], summary, scheduleRows, actualCount));
  const [backend, setBackend] = useState({ status: "checking", model: null, reason: null });
  const [running, setRunning] = useState(false);
  const [sessionId] = useState(() => {
    const suffix = globalThis.crypto?.randomUUID?.().replaceAll("-", "") ?? `${Date.now()}`;
    return `web-${suffix}`.slice(0, 128);
  });

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/health", { signal: controller.signal, headers: { Accept: "application/json" } })
      .then(async (response) => {
        if (!response.ok) throw new Error("health unavailable");
        const payload = await response.json();
        setBackend({
          status: payload.status === "available" ? "available" : "unavailable",
          model: payload.model ?? null,
          reason: payload.reason ?? null,
        });
      })
      .catch((error) => {
        if (error.name !== "AbortError") setBackend({ status: "unavailable", model: null, reason: "LOCAL_AGENT_NOT_RUNNING" });
      });
    return () => controller.abort();
  }, []);

  const submit = async (event) => {
    event?.preventDefault();
    if (!request.trim() || running) return;
    if (backend.status !== "available") {
      setResult({ ...interpretRequest(request, summary, scheduleRows, actualCount), source: "local" });
      return;
    }
    setRunning(true);
    setResult({
      intent: "Agent 正在处理",
      answer: "正在通过本地受限 AgentLoop 检查请求和可用工具……",
      next: "完成前不会生成或下发设备控制指令。",
      source: "agent",
    });
    try {
      const response = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ message: request.trim(), session_id: sessionId }),
      });
      const payload = await response.json();
      if (!response.ok || payload.status !== "completed") {
        setResult({
          intent: "Agent 安全终止",
          answer: "本次请求没有产生可用结果，也没有执行任何设备操作。",
          next: `原因：${payload.failure_code ?? payload.error ?? "AGENT_REQUEST_FAILED"}`,
          source: "agent",
        });
      } else {
        const toolText = payload.tool_calls?.length ? payload.tool_calls.join(" → ") : "未调用业务工具";
        setResult({
          intent: "DeepSeek Agent 已完成",
          answer: payload.final_response,
          next: `本轮 ${payload.steps} 个步骤；工具链：${toolText}。结果仍为仿真、不可执行。`,
          source: "agent",
        });
      }
    } catch (_error) {
      setResult({
        intent: "本地服务不可用",
        answer: "工作台无法连接本地 Agent 服务，没有调用模型或执行业务工具。",
        next: "确认本地服务仍在运行，然后重试。",
        source: "agent",
      });
    } finally {
      setRunning(false);
    }
  };

  const backendLabel = backend.status === "available"
    ? `Agent 已连接 · ${backend.model ?? "DeepSeek"}`
    : backend.status === "checking" ? "正在检查本地 Agent" : "本地解释模式 · Agent 未连接";
  return <section className="agent-console" aria-labelledby="agent-console-title">
    <div className="agent-console-copy"><span className="eyebrow">自然语言入口</span><h2 id="agent-console-title">直接说你想知道什么</h2>
      <p>服务可用时由个人 DeepSeek API 和五个白名单工具处理；否则使用本地只读解释。两种模式都不会操作设备。</p>
      <span className={`agent-connection ${backend.status}`}><i aria-hidden="true" />{backendLabel}</span></div>
    <form className="agent-form" onSubmit={submit}><label htmlFor="energy-question">你的问题</label>
      <div className="agent-input-row"><textarea id="energy-question" value={request} rows={2} disabled={running} onChange={(e) => setRequest(e.target.value)} placeholder="例如：今天的调度是否可以执行？" /><Button type="submit" disabled={running}>{running ? "处理中…" : backend.status === "available" ? "发送给 Agent" : "本地解释"}</Button></div>
      <div className="prompt-presets" aria-label="常用问题">{presets.map((preset) => <button key={preset} type="button" disabled={running} onClick={() => { setRequest(preset); setResult({ ...interpretRequest(preset, summary, scheduleRows, actualCount), source: "local" }); }}>{preset}</button>)}</div>
    </form>
    {result && <div className="agent-answer" role="status"><span className="agent-intent">{result.source === "agent" ? "Agent：" : "识别为："}{result.intent}</span><strong>{result.answer}</strong><p><span>下一步：</span>{result.next}</p></div>}
  </section>;
}

const architectureSteps = [
  { number: "01", title: "自然语言请求", detail: "非技术用户直接描述目标、异常或需要比较的方案。" },
  { number: "02", title: "意图与权限约束", detail: "AgentLoop 只允许选择五个业务白名单工具，不开放 Shell、任意文件或设备控制。" },
  { number: "03", title: "复合调度工具", detail: "create_verified_dispatch 强制串联数据质量检查、优化器和结果校验。" },
  { number: "04", title: "Verifier 与 Evidence", detail: "独立核验功率平衡、储能约束、目标值与证据绑定，并生成完整 Trace。" },
  { number: "05", title: "解释与人工审批", detail: "页面展示结论、限制和下一步；当前只生成建议，不直接操作设备。" },
];

const deliveryStages = [
  ["A", "光伏物理建模", "屋面、立面、停车场子阵列", "complete"],
  ["B", "负荷与场景", "分表聚合、数据质量与基准场景", "complete"],
  ["C", "Agent 运行层", "DeepSeek API、受限 AgentLoop、白名单工具", "complete"],
  ["D", "可验证调度", "优化、Verifier、Evidence、Trace", "complete"],
  ["F", "人机决策界面", "自然语言入口、计划展示与重调度骨架", "complete"],
  ["E", "真实闭环", "预测、动态碳因子与计划—实际遥测", "next"],
];

function Showcase({ summaryRows, scheduleRows, summary }) {
  const pvPeak = Math.max(...scheduleRows.map((row) => Number(row.plannedPvKw) || 0));
  return <>
    <section className="showcase-intro" aria-labelledby="showcase-title">
      <div className="showcase-copy">
        <span className="eyebrow">Portfolio prototype · Energy AI</span>
        <h2 id="showcase-title">把优化模型变成普通人也能使用的能源决策 Agent</h2>
        <p>这个原型把校园负荷、天气驱动光伏、储能优化和大模型交互放进同一个受控闭环。模型负责理解问题和选择工具，确定性程序负责计算、验证与留痕。</p>
        <div className="showcase-tags" aria-label="项目能力标签">
          <span>DeepSeek API</span><span>Guarded AgentLoop</span><span>MILP / HiGHS</span><span>Verifier</span><span>Evidence & Trace</span>
        </div>
      </div>
      <aside className="showcase-boundary" aria-label="原型安全边界">
        <span>当前原型</span><strong>建议型 Agent</strong>
        <p>仿真数据与场景参数用于流程验证；没有设备控制工具，所有计划仍需人工审批。</p>
      </aside>
    </section>

    <Section id="showcase-snapshot" title="演示快照"><div className="showcase-metrics">
      <MetricCard id="showcase-cost" queryId="dispatch_summary" title="单日参考费用" value={cny(summary.totalCostCny)} comparison="经济调度目标" deltaTone="neutral" displayRows={summaryRows} sourceRows={summaryRows} />
      <MetricCard id="showcase-peak" queryId="dispatch_summary" title="峰值购电" value={`${numberCell(summary.peakImportKw)} kW`} comparison="24 小时计划" deltaTone="neutral" displayRows={summaryRows} sourceRows={summaryRows} />
      <MetricCard id="showcase-pv" queryId="dispatch_schedule" title="光伏计划峰值" value={`${numberCell(pvPeak)} kW`} comparison="天气驱动场景" deltaTone="positive" displayRows={scheduleRows} sourceRows={scheduleRows} />
      <MetricCard id="showcase-verifier" queryId="dispatch_summary" title="独立校验" value={summary.verificationPassed ? "通过" : "未通过"} comparison="不可执行 · 待审批" deltaTone={summary.verificationPassed ? "positive" : "negative"} displayRows={summaryRows} sourceRows={summaryRows} />
    </div></Section>

    <Section id="showcase-architecture" title="受控 Agent 闭环">
      <div className="architecture-flow" role="list" aria-label="从自然语言到人工审批的五步闭环">
        {architectureSteps.map((step, index) => <React.Fragment key={step.number}>
          <article className="architecture-step" role="listitem"><span>{step.number}</span><strong>{step.title}</strong><p>{step.detail}</p></article>
          {index < architectureSteps.length - 1 && <span className="architecture-arrow" aria-hidden="true">→</span>}
        </React.Fragment>)}
      </div>
    </Section>

    <Section id="showcase-delivery" title="项目完成度" columns={2}>
      <div className="stage-list" aria-label="项目阶段">
        {deliveryStages.map(([stage, title, detail, state]) => <div className={`stage-item ${state}`} key={stage}>
          <span className="stage-code">{stage}</span><div><strong>{title}</strong><p>{detail}</p></div><span className="stage-state">{state === "complete" ? "已完成" : "下一阶段"}</span>
        </div>)}
      </div>
      <div className="demo-path"><span className="eyebrow">三分钟演示路径</span><h3>从问题，到计划，再到证据</h3>
        <ol><li><span>1</span><div><strong>提出问题</strong><p>在“调度计划”页用自然语言询问计划是否可执行，观察 Agent 如何选择白名单工具。</p></div></li>
          <li><span>2</span><div><strong>查看调度</strong><p>检查负荷、光伏、购电和储能的 24 小时协同计划，以及关键结果指标。</p></div></li>
          <li><span>3</span><div><strong>追溯证据</strong><p>进入“证据与审计”，核对数据质量、Verifier 结果、工具调用和运行 Trace。</p></div></li></ol>
        <div className="ownership-note"><strong>核心实现</strong><p>Agent Harness、白名单工具、优化与验证编排、Evidence/Trace、自然语言结果解释和前端展示。</p></div>
      </div>
    </Section>
  </>;
}

function Overview({ summaryRows, scheduleRows }) {
  const summary = summaryRows[0];
  return <>
    <NaturalLanguageConsole summary={summary} scheduleRows={scheduleRows} actualCount={scheduleRows.filter((row) => Number.isFinite(row.actualLoadKw)).length} />
    <Section id="decision-summary" title="今天的调度结论"><div className="metric-grid">
      <MetricCard id="plan-status" queryId="dispatch_summary" title="计划状态" value={summary.verificationPassed ? "已验证" : "未通过"} comparison="仿真 · 待审批" deltaTone="neutral" displayRows={summaryRows} sourceRows={summaryRows} />
      <MetricCard id="total-cost" queryId="dispatch_summary" title="参考费用" value={cny(summary.totalCostCny)} comparison="电价未现场核验" deltaTone="neutral" displayRows={summaryRows} sourceRows={summaryRows} />
      <MetricCard id="peak-import" queryId="dispatch_summary" title="峰值购电" value={`${numberCell(summary.peakImportKw)} kW`} comparison="单日仿真" deltaTone="neutral" displayRows={summaryRows} sourceRows={summaryRows} />
      <MetricCard id="pv-utilization" queryId="dispatch_summary" title="光伏自用率" value={percent(summary.pvSelfConsumptionRatio)} comparison="无弃光" deltaTone="positive" displayRows={summaryRows} sourceRows={summaryRows} />
      <MetricCard id="battery-throughput" queryId="dispatch_summary" title="储能吞吐量" value={`${numberCell(summary.batteryThroughputKwh)} kWh`} comparison="150 kW / 400 kWh 假设" deltaTone="neutral" displayRows={summaryRows} sourceRows={summaryRows} />
    </div></Section>
    <Section id="power-plan" title="24 小时调度计划" columns={2}>
      <EvidenceChart id="power-plan-chart" queryId="dispatch_schedule" title="负荷、光伏与购电" description="逐时计划值；实际数据未混入该图。" spec={powerPlanSpec} rows={scheduleRows} sourceRows={scheduleRows} height={290} variant="card" />
      <EvidenceChart id="storage-plan-chart" queryId="dispatch_schedule" title="储能充放电与电量" description="充电、放电功率与计划储能电量；设备参数仍待现场核验。" spec={storagePlanSpec} rows={scheduleRows} sourceRows={scheduleRows} height={290} variant="card" />
    </Section>
    <Section id="hourly-plan-table" title="逐时计划明细"><DataComponent id="hourly-plan" queryId="dispatch_schedule" kind="table" variant="card" title="计划表" description="所有数值均来自同一份已验证调度产物。" displayRows={scheduleRows} sourceRows={scheduleRows}>
      <DataTable rows={scheduleRows} columns={planColumns} rowKey="slot" />
    </DataComponent></Section>
  </>;
}

function Monitoring({ scheduleRows, rules }) {
  const actualRows = scheduleRows.filter((row) => Number.isFinite(row.actualLoadKw));
  const coverage = scheduleRows.length ? actualRows.length / scheduleRows.length : null;
  return <>
    <section className="availability-banner" role="status"><div><span className="eyebrow">实际遥测状态</span><h2>尚未接入计划—实际闭环</h2></div><p>当前有 {actualRows.length}/{scheduleRows.length} 个时段包含实际负荷。所有实际值保持为空，不会把缺失数据当成 0。</p></section>
    <Section id="monitoring-overview" title="偏差监测"><div className="monitoring-grid">
      <MetricCard id="actual-coverage" queryId="dispatch_schedule" title="实际数据覆盖率" value={coverage == null ? "—" : percent(coverage)} comparison={`${actualRows.length}/${scheduleRows.length} 个时段`} deltaTone="negative" displayRows={scheduleRows} sourceRows={scheduleRows} />
      <MetricCard id="deviation-status" queryId="dispatch_schedule" title="偏差结论" value={actualRows.length ? "待计算" : "无法计算"} comparison="等待实际遥测" deltaTone="neutral" displayRows={scheduleRows} sourceRows={scheduleRows} />
      <MetricCard id="reschedule-status" queryId="monitoring_rules" title="重调度建议" value="未触发" comparison="规则草案 · 非执行指令" deltaTone="neutral" displayRows={rules} sourceRows={rules} />
    </div></Section>
    <Section id="plan-actual-view" title="计划与实际对照">
      <EvidenceChart id="plan-actual-chart" queryId="dispatch_schedule" title="购电功率对照" description="实际购电尚未接入，因此图中只显示计划线；不会补零或插值。" spec={deviationSpec} rows={scheduleRows} sourceRows={scheduleRows} height={290} variant="card" />
      <DataComponent id="plan-actual-table" queryId="dispatch_schedule" kind="table" variant="card" title="逐时偏差明细" description="接入实际遥测后，将在相同时区与时段粒度上计算偏差。" displayRows={scheduleRows} sourceRows={scheduleRows}><DataTable rows={scheduleRows} columns={monitoringColumns} rowKey="slot" /></DataComponent>
    </Section>
    <Section id="reschedule-framework" title="重新调度建议骨架" columns={2}>
      <DataComponent id="reschedule-rules" queryId="monitoring_rules" kind="table" variant="card" title="触发规则草案" description="阈值为待评审策略，不是经过现场验证的控制参数。" displayRows={rules} sourceRows={rules}><DataTable rows={rules} columns={ruleColumns} rowKey="ruleId" searchable={false} /></DataComponent>
      <DataComponent id="reschedule-decision" queryId="monitoring_rules" kind="custom" variant="card" title="当前建议" description="建议生成必须基于真实偏差、数据质量与完整验证链。" displayRows={rules} sourceRows={rules}>
        <div className="recommendation-empty"><span className="recommendation-icon" aria-hidden="true">↻</span><strong>等待实际遥测，暂不生成重调度建议</strong><p>未来触发后仍只生成“建议”：先重新检查数据质量，再调用优化器、Verifier、Evidence 和 Trace，最后等待人工审批。</p><ol><li>检测连续偏差或覆盖率不足</li><li>冻结输入版本并生成新候选计划</li><li>独立验证约束、费用和证据绑定</li><li>展示新旧计划差异，交给人员决定</li></ol></div>
      </DataComponent>
    </Section>
  </>;
}

function Audit({ summaryRows, findings, tools, trace }) {
  const summary = summaryRows[0];
  return <>
    <section className="audit-hero"><div><span className="eyebrow">可追溯运行记录</span><h2>每个结论都能回到工具、证据和产物</h2></div><dl><div><dt>调度运行</dt><dd>{summary.dispatchRunId}</dd></div><div><dt>计划编号</dt><dd>{summary.scheduleId}</dd></div><div><dt>求解器</dt><dd>{summary.solver} · {summary.solverStatus}</dd></div></dl></section>
    <Section id="quality-audit" title="数据质量与限制"><DataComponent id="quality-findings" queryId="quality_findings" kind="table" variant="card" title="质量检查结果" description="这些限制决定计划只能作为仿真和方案比较。" displayRows={findings} sourceRows={findings}><DataTable rows={findings} columns={qualityColumns} rowKey="findingId" /></DataComponent></Section>
    <Section id="agent-audit" title="Agent 白名单与执行轨迹" columns={2}>
      <DataComponent id="whitelist-tools" queryId="agent_tools" kind="table" variant="card" title="五个白名单工具" description="自然语言只能选择这些受约束能力；没有设备控制工具。" displayRows={tools} sourceRows={tools}><DataTable rows={tools} columns={toolColumns} rowKey="tool" searchable={false} /></DataComponent>
      <DataComponent id="agent-trace" queryId="agent_trace" kind="table" variant="card" title="最近一次运行轨迹" description="按顺序记录请求解析、工具调用、验证和结果生成。" displayRows={trace} sourceRows={trace}><DataTable rows={trace} columns={traceColumns} rowKey="sequence" /></DataComponent>
    </Section>
  </>;
}

export function DashboardContent() {
  const { activeTabId } = useDashboardTabs([
    { id: "showcase", label: "项目概览" }, { id: "overview", label: "调度计划" }, { id: "monitoring", label: "偏差与重调度" }, { id: "audit", label: "证据与审计" },
  ]);
  const { reviewedRows } = useDataApp();
  const summaryRows = reviewedRows("dispatch_summary");
  const scheduleRows = reviewedRows("dispatch_schedule", ["slot"]);
  const findings = reviewedRows("quality_findings", ["findingId"]);
  const rules = reviewedRows("monitoring_rules", ["ruleId"]);
  const tools = reviewedRows("agent_tools", ["tool"]);
  const trace = reviewedRows("agent_trace", ["sequence"]);
  const summary = summaryRows[0] ?? {};
  const tab = activeTabId ?? "showcase";
  const subtitle = useMemo(() => summary.studyDay ? `${summary.studyDay} · ${summary.scenarioName ?? "校园光储负荷调度"}` : "已验证调度的只读决策界面", [summary.studyDay, summary.scenarioName]);
  return <article className="page energyops-page">
    <header className="energyops-hero"><div><span className="eyebrow">Campus EnergyOps Agent</span><h1>{tab === "showcase" ? "可验证的校园光储调度原型" : "从自然语言到可解释调度"}</h1><p>{tab === "showcase" ? "面向项目展示的交互原型：受控 Agent、确定性优化、独立验证与完整证据链。" : subtitle}</p></div><div className="hero-mode"><span>当前模式</span><strong>{tab === "showcase" ? "展示原型" : "只读仿真"}</strong></div></header>
    <StatusStrip summary={summary} />
    {tab === "showcase" && <Showcase summaryRows={summaryRows} scheduleRows={scheduleRows} summary={summary} />}
    {tab === "overview" && <Overview summaryRows={summaryRows} scheduleRows={scheduleRows} />}
    {tab === "monitoring" && <Monitoring scheduleRows={scheduleRows} rules={rules} />}
    {tab === "audit" && <Audit summaryRows={summaryRows} findings={findings} tools={tools} trace={trace} />}
    <footer className="energyops-footer"><strong>安全边界</strong><span>本工作台仅展示仿真、解释证据和生成建议；不连接设备控制，不自动执行调度。</span></footer>
  </article>;
}
