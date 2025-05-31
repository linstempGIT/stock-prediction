# 股票预测系统

该项目使用深度学习方法（LSTM和神经网络）进行股票价格预测，包含完整的数据处理、模型训练、预测和可视化流程。

## 功能特性

- 股票数据获取与预处理
- LSTM Attention模型训练与预测
- 神经网络模型训练与预测
- 预测结果可视化
- 中文支持（字体处理）

## 安装指南

### 前置要求
- Python 3.8+
- [UV](https://github.com/astral-sh/uv) (推荐) 或 pip

### 安装步骤
```bash
# 创建虚拟环境
uv venv
source .venv/bin/activate

# 安装依赖
uv pip install -e .
```

## 使用说明

### 1. 配置设置
复制示例配置文件并修改：
```bash
cp configs/app-config.example.yml configs/app-config.yml
```

### 2. 数据处理
```python
from src.data_process.commonDataProcess import process_data

# 处理股票数据
process_data(stock_code="000001.SZ", start_date="2020-01-01")
```

### 3. 模型训练
训练LSTM模型：
```bash
python src/train/trainLSTMAttention.py
```

训练神经网络模型：
```bash
python src/train/trainNN.py
```

### 4. 运行预测
```bash
python src/predict/predict_stock.py
```

### 5. 可视化结果
```python
from src.visualize.dailyVisual import visualize_stock

# 可视化股票数据
visualize_stock(stock_code="600519.SH")
```

## 项目结构

```
stock-prediction/
├── configs/          # 配置文件
│   └── app-config.example.yml
├── data/             # 数据目录
│   ├── baseInfo/     # 股票基本信息
│   ├── dailyInfo/    # 每日行情数据
│   ├── models/       # 训练好的模型
│   ├── processed/    # 处理后的数据
│   └── visualImage/  # 可视化图像
├── src/              # 源代码
│   ├── data_process/ # 数据处理
│   ├── models/       # 模型实现
│   ├── predict/      # 预测功能
│   ├── train/        # 训练脚本
│   ├── utils/        # 工具函数
│   └── visualize/    # 可视化
├── tests/            # 单元测试
└── pyproject.toml    # 项目配置
```

## 贡献指南

欢迎通过Issue或Pull Request贡献代码：
1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -am 'Add some feature'`)
4. 推送分支 (`git push origin feature/your-feature`)
5. 创建Pull Request

## 许可证
[MIT License](LICENSE) (请添加LICENSE文件)
