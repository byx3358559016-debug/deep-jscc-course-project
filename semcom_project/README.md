# 轻量级 Deep JSCC 图像语义通信课程设计

本项目在教师提供的 `basic_code` 基础上实现了可复现、无训练/测试泄漏的图像语义通信实验。系统由卷积编码器、可微无线信道和卷积解码器组成，并加入残差块、逐样本发送功率归一化、潜变量通道数消融及固定/随机 SNR 鲁棒训练对比。

## 1. 已完成内容

- 原样运行教师 `train.py`、`evaluate.py`，记录原始复现结果；
- 将 24 张 Kodak 图片按图像级划分为 16 张训练、4 张验证、4 张测试；
- 训练阶段只从训练图片随机裁剪 `64×64` patch，测试图片不会进入训练；
- 实现 AWGN 和慢 Rayleigh 衰落信道；
- 实现逐样本发送功率归一化和轻量残差块；
- 比较 latent channel 为 8、16、32 时的压缩率与图像质量；
- 比较固定 5 dB 与 `{-10,-5,0,5,10}` dB 随机 SNR 训练；
- 每个测试 SNR、每张测试图重复 5 次独立噪声，实现均值和标准差统计；
- 自动保存 MSE、PSNR、SSIM、训练日志、重建图、性能曲线和复杂度结果；
- 提供 5 个单元测试和一键复现实验脚本。

## 2. 环境配置

本次实际实验环境：

- Windows，Python 3.13.5；
- PyTorch 2.9.0 + CUDA 13.0；
- NVIDIA GeForce RTX 5060 Ti 16 GB；
- NumPy 2.1.3、Pillow、Matplotlib 3.10、pytest。

创建独立环境后安装依赖：

```powershell
cd semcom_project
pip install -r requirements.txt
```

程序自动选择 CUDA；没有 GPU 时会回退到 CPU。也可显式添加 `--device cpu`。

## 3. 数据集路径与划分

默认数据集路径为：

```text
../basic_code/kodak/
```

固定划分保存在 `configs/split.json`，随机种子为 42：

- 训练集 16 张；
- 验证集 4 张；
- 测试集 4 张；
- 三个子集没有文件重叠。

训练集每个 epoch 动态生成 256 个随机 patch，并随机水平翻转。验证和测试直接使用未参与训练的完整分辨率图片。

## 4. 运行命令

### 4.1 运行测试

```powershell
python -m pytest -q
```

预期输出为 `5 passed`。

### 4.2 一键复现五组实验

```powershell
python run_all.py --force --epochs 80 --samples-per-epoch 256 --repeats 5 --device auto
python benchmark.py
```

如果 checkpoint 已存在并且不希望重新训练，可去掉 `--force`。

### 4.3 单独训练

规范基线：

```powershell
python train.py --name baseline_c16_random --variant baseline --latent-channels 16 --train-snrs -10 -5 0 5 10
```

改进 C16 随机 SNR 模型：

```powershell
python train.py --name improved_c16_random --variant improved --latent-channels 16 --train-snrs -10 -5 0 5 10
```

固定 5 dB 模型：

```powershell
python train.py --name improved_c16_fixed5 --variant improved --latent-channels 16 --train-snrs 5
```

### 4.4 单独评估和绘图

```powershell
python evaluate.py --checkpoint outputs/checkpoints/improved_c16_random.pth --snrs -10 -5 0 5 10 --repeats 5
python plot_results.py
```

## 5. 主要参数

| 参数 | 设置 |
|---|---|
| 输入 patch | `64×64` RGB |
| Batch size | 16 |
| 最大 epoch | 80 |
| 每个 epoch 样本 | 256 个随机 patch |
| 优化器 | Adam |
| 学习率 | `1e-3` |
| 损失 | MSE |
| 随机训练 SNR | `-10,-5,0,5,10 dB` |
| 测试 SNR | `-10,-5,0,5,10 dB` |
| 验证频率 | 每 5 epoch |
| 随机种子 | 42 |

当编码器经历两次 2 倍下采样、两个实数表示一个复数信道符号时，带宽比为 `k/n=C/96`。因此 C8、C16、C32 分别对应约 `1/12`、`1/6`、`1/3`。

## 6. 核心结果

以下结果来自严格隔离的 4 张测试图，每个 SNR 共 20 个样本（4 张图片×5 次独立噪声）。

### 6.1 5 dB 对比

| 模型 | k/n | 参数量 | PSNR/dB | SSIM | MSE |
|---|---:|---:|---:|---:|---:|
| Baseline C16 随机 SNR | 1/6 | 72,243 | 21.48 | 0.5324 | 0.00732 |
| Improved C8 随机 SNR | 1/12 | 65,579 | 22.98 | 0.6252 | 0.00526 |
| Improved C16 随机 SNR | 1/6 | 81,779 | 23.96 | 0.6740 | 0.00416 |
| Improved C32 随机 SNR | 1/3 | 128,003 | 24.69 | 0.7135 | 0.00351 |
| Improved C16 固定 5 dB | 1/6 | 81,779 | 26.35 | 0.7500 | 0.00247 |

在相同 C16 和随机 SNR 条件下，改进模型比规范基线提升约 2.48 dB PSNR、0.142 SSIM，MSE 下降约 43.2%。

### 6.2 随机 SNR 与固定 SNR

固定 5 dB 模型在匹配条件下达到 26.35 dB，但在 -10 dB 时只有 14.03 dB。随机 SNR 的 C16 模型在 -10 dB 仍有 20.84 dB，说明随机 SNR 训练牺牲部分高 SNR 峰值性能，换取了显著的低 SNR 鲁棒性。

### 6.3 教师原始代码复现

教师代码在 24 张 Kodak 图片上同时训练和测试，得到 -10 至 10 dB 下 PSNR 为 19.34、21.97、23.43、24.02、24.27 dB。该结果用于确认代码可运行，但因为存在数据泄漏，不与严格测试集结果作直接优劣比较。

## 7. 输出目录

```text
outputs/
├── checkpoints/       # 本地训练生成；GitHub 版本不包含权重
├── logs/              # 本地训练生成；GitHub 版本不包含日志
├── metrics/           # 原始样本、均值/标准差和复杂度 CSV
├── curves/            # SNR 曲线、训练曲线和系统结构图
└── reconstructions/   # 原图与五个 SNR 重建图
```

关键文件：

- `outputs/metrics/model_comparison.csv`：所有模型和 SNR 的汇总结果；
- `outputs/metrics/complexity.csv`：参数量、权重大小、CPU/GPU 推理时间；
- `outputs/curves/snr_psnr.png`；
- `outputs/curves/snr_ssim.png`；
- `outputs/reconstructions/*_comparison.png`。

## 8. 代码模块

- `data.py`：数据划分读取、训练 patch 和完整测试图加载；
- `model.py`：编码器、残差块、功率归一化、AWGN/Rayleigh 信道和解码器；
- `train.py`：训练、验证、early stopping、checkpoint 与日志保存；
- `evaluate.py`：重复信道采样、MSE/PSNR/SSIM、CSV 和重建图；
- `plot_results.py`：性能曲线、训练曲线和结构图；
- `benchmark.py`：CPU/GPU 推理时间与模型大小；
- `run_all.py`：五组核心实验的一键入口；
- `tests/test_core.py`：数据划分、形状、功率、SNR 和带宽比测试。

## 9. 可复现性说明

训练、数据裁剪、模型初始化和评估噪声均由固定种子控制。验证信道使用独立且固定的随机状态，避免验证过程改变后续训练随机数。checkpoint 同时保存网络权重、数据路径、训练 SNR、信道类型、参数量、带宽比、最佳 epoch、训练时间和 PyTorch 版本。

不同显卡、CUDA/cuDNN 版本及线程设置可能造成轻微数值差异；应关注曲线趋势和统计结果，而不要求最后一位小数完全一致。

## 10. 注意事项

- GitHub 提交版本不包含教师原始权重或改进模型权重，需要运行训练命令在本地生成；
- 不要将教师原代码的同集训练/测试结果描述成泛化性能；
- Kodak 只有 24 张图片，本实验属于课程级验证，不能据此宣称在大规模自然图像上充分泛化；
- 如更换数据集，需要同步修改 split 文件并确保训练、验证、测试集仍然互斥。
