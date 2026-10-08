# A2C 模型权重 - 占位符

> ⏳ **此目录待上传 A2C 模型权重**

## 预期结构

```
rl_models/a2c/
├── README.md                                      # 本文件
│
├── period1_checkpoints/                          # Period 1 (2011-2015) 模型
│   ├── a2c_Commodity_period1.pt
│   ├── a2c_Equity_Index_period1.pt
│   ├── a2c_Fixed_Income_period1.pt
│   ├── a2c_Forex_period1.pt
│   └── a2c_All_period1.pt
│
├── period2_checkpoints/                          # Period 2 (2016-2019) 模型
│   ├── a2c_Commodity_period2.pt
│   ├── a2c_Equity_Index_period2.pt
│   ├── a2c_Fixed_Income_period2.pt
│   ├── a2c_Forex_period2.pt
│   └── a2c_All_period2.pt
│
└── results/                                        # A2C 评估结果
    ├── a2c_results_wide.csv                       # 按资产和日期组织的累积收益
    └── metrics_report.csv                         # 指标对比报告
```

## 模型文件说明

| 文件 | 说明 |
|------|------|
| `a2c_*_period1.pt` | Period 1 (2011-2015) 期间训练的 A2C 模型权重 |
| `a2c_*_period2.pt` | Period 2 (2016-2019) 期间训练的 A2C 模型权重 |
| `a2c_results_wide.csv` | 每个资产类别的累积收益数据（按日期索引） |

## CSV 文件格式

`a2c_results_wide.csv` 应包含以下列：

```
date (index), Commodity, Equity Index, Fixed Income, Forex, All
2011-01-01,    1.0,      1.0,          1.0,         1.0,    1.0
2011-01-02,    1.002,    1.005,        0.998,       1.001,  1.001
...
2019-12-31,    1.234,    1.567,        0.987,       1.123,  1.228
```

- **索引**: 日期 (YYYY-MM-DD 格式)
- **列**: 各资产类别的累积财富路径 (W_t = 1 + cumsum(daily_returns))

## 模型架构要求

A2C 模型应包含：

1. **Actor 网络** - 生成头寸权重 (0 ~ 1)
2. **Critic 网络** - 评估状态价值

3. **输入状态** (9-dim):
   - 近期收益 (5 日)
   - 波动率 (realized vol)
   - 相对强度指标 (RSI)
   - 等

4. **输出**:
   - Actor: 头寸权重分布
   - Critic: 状态值估计

## 上传说明

将以下内容提交到此目录：

1. ✅ 模型权重 (`.pt` 文件)
2. ✅ 评估结果 CSV
3. ✅ 模型配置文档 (config.json)
4. ✅ 训练日志 (可选)

## 相关代码位置

- **模型实现**: `src/core/models/a2c/`
- **训练脚本**: `src/scripts/run_a2c_model.py`
- **测试脚本**: `tests/test_a2c.py`

## 使用示例

```python
from src.core.models.a2c import A2CEvaluator
import torch

# 加载模型
evaluator = A2CEvaluator(
    model_path="rl_models/a2c/period1_checkpoints/",
    data_path="data/",
    asset_class="Commodity"
)

# 评估策略
returns = evaluator.evaluate(
    start_date="2011-01-01",
    end_date="2015-12-31"
)
```

---

**上传截止**: 待定
**负责人**: [待补充]

