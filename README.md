# Deep JSCC 图像语义通信课程设计

本仓库是在课程提供的基础代码上完成的轻量级图像语义通信课程设计。项目实现了卷积编码器、可微 AWGN 信道和卷积解码器，并进一步加入残差结构、PReLU、逐样本功率归一化、潜变量通道数消融以及固定/随机 SNR 鲁棒性对比。

仓库保留课程提交需要的核心代码、运行说明、技术报告、汇总指标和实验图表，不包含模型权重、Kodak 数据集、缓存和训练中间文件。模型权重可按照本文命令重新训练生成。

## 仓库结构

```text
.
├── basic_code/                  # 教师基础 train.py 和 evaluate.py
├── semcom_project/
│   ├── data.py                  # 数据划分、训练 patch 和完整图测试集
│   ├── model.py                 # 基线/改进模型及 AWGN、Rayleigh 信道
│   ├── train.py                 # 训练、验证和最佳 checkpoint 保存
│   ├── evaluate.py              # 多 SNR 重复测试及指标计算
│   ├── plot_results.py          # 结果曲线绘制
│   ├── benchmark.py             # 参数量和 CPU/GPU 推理时间
│   ├── run_all.py               # 五组核心实验的一键入口
│   ├── configs/split.json       # 固定 16/4/4 图像级划分
│   ├── tests/test_core.py       # 核心功能测试
│   └── outputs/                 # 汇总 CSV、曲线和重建对比图
└── report/语义通信课程设计技术报告.pdf
```

## 环境配置

建议使用 Python 3.10 或更高版本：

```bash
cd semcom_project
pip install -r requirements.txt
python -m pytest -q
```

程序可在 CPU 上运行，也会在 CUDA 可用时自动使用 GPU。实际课程实验环境及依赖版本见 `semcom_project/README.md` 和 `environment.yml`。

## 数据集准备

本仓库不上传 Kodak 图像。请将课程提供的 24 张 PNG 图像放入：

```text
basic_code/kodak/
```

程序默认读取 `../basic_code/kodak/`。固定划分保存在 `semcom_project/configs/split.json`，其中训练集、验证集和测试集分别包含 16、4、4 张图像，三个集合互不重叠。

## 训练与测试

一键复现五组实验：

```bash
cd semcom_project
python run_all.py --force --epochs 80 --samples-per-epoch 256 --repeats 5 --device auto
python benchmark.py
```

单独训练改进 C16 随机 SNR 模型：

```bash
python train.py --name improved_c16_random --variant improved --latent-channels 16 --train-snrs -10 -5 0 5 10
```

训练完成后测试并绘图：

```bash
python evaluate.py --checkpoint outputs/checkpoints/improved_c16_random.pth --snrs -10 -5 0 5 10 --repeats 5
python plot_results.py
```

没有 GPU 时可将 `--device auto` 改为 `--device cpu`。主要参数、单模型命令和结果解释见 `semcom_project/README.md`。

## 已包含的实验结果

- 不同 SNR 下的 MSE、PSNR、SSIM 汇总 CSV；
- SNR-MSE、SNR-PSNR、SNR-SSIM 曲线；
- 原图与多 SNR 重建图；
- 原始模型与改进模型对比；
- 潜变量通道数和固定/随机 SNR 对比；
- 参数量、模型体积以及 CPU/GPU 推理时间。

完整课程报告见 [`report/语义通信课程设计技术报告.pdf`](report/语义通信课程设计技术报告.pdf)。

## 重要说明

- `*.pth`、`*.pt` 和 `*.ckpt` 已由 `.gitignore` 排除；
- Kodak 数据集目录已排除，需在本地自行放置；
- 已提交的 CSV 和图像是报告所用的实验结果，不依赖仓库中的预训练权重；
- 教师原代码使用同一组 Kodak 图像训练和测试，报告将其仅作为基础流程复现；严格对比实验使用固定的 16/4/4 图像级划分。
