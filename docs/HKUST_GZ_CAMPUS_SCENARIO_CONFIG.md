# 港科广校园场景配置说明

当前正式开发入口使用：

`config/scenarios/hkust_gz_campus_baseline.json`

该配置把真实数据、天气推算和设备假设分开管理：

| 组件 | 当前口径 |
|---|---|
| 校园负荷 | 2026-02-19 港科广 4,499 台完整观测分表聚合 |
| 天气 | 港科广南沙校园附近 NASA POWER 逐小时辐照度、气温和风速 |
| 光伏 | 天气推算，300 kWp 场景假设，不是校园实测 |
| 储能 | 150 kW / 400 kWh 场景假设，不是设备铭牌 |
| 电价 | 九江旧方案广东 2025 参考分时价格，未经校园结算核验 |
| 负荷控制 | 不允许；校园聚合负荷是固定输入 |
| 电网策略 | 允许参考低价充电，禁止上网 |
| 发布策略 | `simulation_only=true`、`executable=false` |

## 首次生成校园天气光伏

```bash
python scripts/prepare_hkust_gz_weather_pv_fixture.py
```

该命令从 NASA POWER 获取配置日期和校园位置的逐小时天气，并生成：

- `data/fixtures/hkust_gz_weather_2026-02-19_60min.csv`
- `data/fixtures/hkust_gz_weather_pv_2026-02-19_60min.csv`
- `data/fixtures/hkust_gz_weather_pv_manifest.json`
- `data/fixtures/hkust_gz_weather_pv_quality.json`

## 运行校园调度

```bash
python scripts/run_hkust_harness_workflow.py
```

默认结果写入：

`outputs/hkust_gz_campus_harness/`

其中 `scenario_config.json` 是该次运行实际采用的配置快照，`request.json` 是优化器收到的严格请求，`schedule.json` 是计划，`verification.json` 是独立校验。

## 最常修改的参数

编辑 `config/scenarios/hkust_gz_campus_baseline.json`：

```json
{
  "pv": {
    "capacity_kwp": 300.0
  },
  "battery": {
    "capacity_kwh": 400.0,
    "initial_energy_kwh": 200.0,
    "min_energy_kwh": 40.0,
    "max_energy_kwh": 360.0,
    "max_charge_kw": 150.0,
    "max_discharge_kw": 150.0
  },
  "required_terminal_energy_kwh": 200.0
}
```

修改光伏容量后必须重新运行天气光伏生成脚本。修改储能参数后只需重新运行调度入口。

容量约束必须一致：

```text
0 <= min_energy <= initial_energy <= max_energy <= capacity
required_terminal_energy <= max_energy
```

## 当前容量情景

- small：75 kW / 200 kWh；
- medium：150 kW / 400 kWh；
- large：300 kW / 800 kWh。

这些方案只用于敏感性分析，不是采购或施工建议。获得校园光伏、储能铭牌和正式电价后，应替换配置中的假设值并保留新的配置版本，不要覆盖旧运行证据。

## 规划物理光伏子阵列

阶段 A 的配置为：

`config/hkust_gz_physical_pv.json`

它把规划光伏拆成三个可独立审计的子阵列：

| 子阵列 | 组件数量 | 装机容量 | 当前方向口径 |
|---|---:|---:|---|
| 屋面 | 15,924 | 10,350.6 kWp | 倾角 10°、正南 |
| 立面 | 14,496 | 9,422.4 kWp | 倾角 90°、东南西北等分 |
| 停车场 | 468 | 304.2 kWp | 倾角 10°、正南 |

生成曲线：

```bash
python scripts/build_hkust_gz_physical_pv.py
```

输出包括屋面、立面、停车场以及立面四个方向的逐时功率，并为屋面、屋面加停车场、全量极限建设三个场景分别生成 CSV、manifest 和质量报告。组件数量和 650 Wp 单块功率来自规划资料；方向、温度参数、逆变器和系统损失仍是待核验假设。规划资料中的年发电量不进入计算。

## 阶段 B：容量与双目标分析

运行：

```bash
python scripts/run_stage_b_analysis.py
```

脚本固定使用研究日 4,499 台完整观测分表聚合负荷，并遍历：

- 三个规划光伏场景；
- 75 kW / 200 kWh、150 kW / 400 kWh、300 kW / 800 kWh 三档储能；
- 经济、低碳、成本约束低碳三种目标。

因此一次生成 27 个计划，并为每个计划保存 schedule 和独立 verification。汇总输出位于 `outputs/hkust_gz_stage_b/`。

当前无储能弃光率从屋面情景的 38.07% 增至全量光伏情景的 55.76%。在全量光伏情景下，300 kW / 800 kWh 储能仍只能把弃光率降至约 54.53%。在禁止上网且只使用单日负荷的条件下，这些结果用于暴露容量不匹配和验证算法，不能用于确定采购容量。

阶段 B 的碳因子仍是版本化合成曲线，电价仍未经校园结算核验。正式预测、动态碳因子、电价和并网规则交付后，应通过 Provider/配置替换输入，并重新生成完整证据，不覆盖本次仿真结果。
