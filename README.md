# Campus EnergyOps Agent

一个面向高校园区的可验证光储调度与自然语言决策原型。

项目将天气驱动的物理光伏模型、储能经济/低碳优化、独立 Verifier、
Evidence Pack 和受控 Agent Loop 组合成一条可审计流程。DeepSeek API 只负责
理解请求、选择白名单工具和解释已验证证据；所有能源计算由确定性 Python
程序完成。

> 公开仓库只包含合成负荷、公开天气衍生数据和规划仿真场景，不包含真实
> 电表 ID、楼栋/房间地址、逐表负荷、API 密钥或设备控制接口。

## 核心能力

- 逐时太阳位置、Erbs 辐照度分解、阵列平面辐照度、NOCT 温度修正、逆变器
  效率和系统损失组成的物理光伏链路；
- CVXPY + HiGHS 实现经济、低碳和成本约束低碳三类调度目标；
- 独立 Verifier 复算功率平衡、SOC、费用、碳排、弃光和储能吞吐；
- 自有 `AgentLoop`、`ToolRegistry`、`PolicyGate`、Session 和脱敏 Trace；
- 五个业务白名单工具，以及强制封装质量门、优化、验证和证据的
  `create_verified_dispatch`；
- React 决策台展示调度计划、偏差与重调度骨架、Evidence 和审计信息。

## 系统架构

```text
自然语言请求
      │
      ▼
EnergyOps Agent Harness ── DeepSeek API
Session / Loop / Policy / Trace
      │ 仅允许白名单 Tool Call
      ▼
数据质量门 → 物理光伏 → MILP 调度 → 独立 Verifier
                                      │
                                      ▼
                              Evidence Pack
                                      │
                                      ▼
                         人工审批前的解释与展示
```

所有调度结果固定为：

```text
simulation_only=true
executable=false
```

## 公开数据说明

公开版使用一条确定性生成的 24 小时合成校园负荷曲线，以便完整运行优化、
Agent 工具和前端演示。天气输入来自逐时公开天气数据，光伏和储能容量均为
规划或敏感性分析假设。

完整的私有案例曾用于验证数据工程流程：处理约 4,879 万条累计电表记录和
4,774 个表计通道，并构建研究日聚合负荷面板。相关表计标识、地址、备注和
逐表数据不进入本仓库。详见 [DATA_PRIVACY.md](DATA_PRIVACY.md)。

## 调度实验

阶段 B 组合：

```text
3 个光伏场景 × 3 档储能规格 × 3 种目标 = 27 个计划
```

三个规划光伏组合为屋面、屋面加停车场、屋面加立面加停车场。三档储能为
75 kW / 200 kWh、150 kW / 400 kWh 和 300 kW / 800 kWh。这些参数只用于
验证算法与业务流程，不构成校园建设或采购建议。

## 五个 Agent 工具

| Tool | 职责 |
|---|---|
| `get_available_scenarios` | 查询允许使用的版本化场景 |
| `get_data_quality_report` | 查看完整性、连续性、单位和质量状态 |
| `create_verified_dispatch` | 执行质量门、优化、Verifier、Evidence 和 Trace |
| `compare_verified_plans` | 比较同版本且验证通过的调度计划 |
| `explain_evidence` | 按 Evidence ID 解释关键数字和限制 |

模型不会获得 shell、任意文件、数据库凭据或设备控制权限。

## 快速开始

需要 Python 3.11+。

```bash
uv sync --extra dev
uv run pytest -q
```

运行 27 组容量与双目标分析：

```bash
uv run python scripts/run_stage_b_analysis.py
```

重新生成公开版已验证调度和工作台数据：

```bash
uv run python scripts/build_public_demo_artifacts.py
uv run python scripts/build_stage_f_dashboard_snapshot.py --sync-dashboard
```

构建并启动前端：

```bash
cd apps/energyops-dashboard
npm ci
npm test
npm run build
cd ../..
uv run python scripts/run_stage_f_dashboard.py
```

浏览器打开 `http://127.0.0.1:4173`。

## 可选 DeepSeek API

离线测试和确定性调度不需要 API Key。启用自然语言 Agent 时，将个人密钥
保存在仓库外的 `~/.config/energyops/secrets.env`：

```text
DEEPSEEK_API_KEY=your-key
```

服务只在本机回环地址开放。密钥不会进入浏览器、静态文件或 Trace。

## 验证

- Python：82 项自动化测试；
- Dashboard：公开数据安全边界测试与 Vite 生产构建；
- CI：每次 push 和 pull request 自动执行 Python 测试与前端构建。

## 技术栈

Python、Pandas、Pydantic、CVXPY、HiGHS、DeepSeek API、React、Vite、Pytest。

## 当前边界

本项目是求职作品集中的仿真决策辅助系统，不控制真实光伏、储能或电网设备，
不声称真实电费节省、真实减排或推荐设备容量。正式预测、动态碳因子、核验后
的电价/并网规则和实时遥测仍属于后续 Provider 接入范围。
