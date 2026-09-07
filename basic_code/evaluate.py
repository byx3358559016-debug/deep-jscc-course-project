import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os

# 导入在 train.py 中定义的类
from train import SemanticCommSystem, KodakDataset


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESOLUTION = 64

    # 多尺度信噪比测试
    TEST_SNRS = [-10, -5, 0, 5, 10]

    # 固定SNR=5测试
    # TEST_SNRS = [5]

    transform = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.ToTensor()
    ])

    # 加载测试数据 (这里直接用 kodak 作为测试集进行演示)
    dataset = KodakDataset(img_dir='./kodak', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # batch_size=1 方便逐张图计算指标

    # 初始化模型并加载之前训练好的权重
    model = SemanticCommSystem().to(device)
    try:
        model.load_state_dict(torch.load('semantic_model.pth'))
        print("成功加载模型权重！")
    except Exception as e:
        print(f"加载模型失败，请确保先运行 train.py。错误信息: {e}")
        return

    model.eval()  # 设置为评估模式，关闭 Dropout 和 BatchNorm 的随机性

    print("-" * 50)
    print(f"{'SNR (dB)':<10} | {'MSE':<10} | {'PSNR (dB)':<10} | {'SSIM':<10}")
    print("-" * 50)

    # 禁用梯度计算以节省显存并加速
    with torch.no_grad():
        for snr in TEST_SNRS:
            total_mse, total_psnr, total_ssim = 0, 0, 0

            for img in dataloader:
                img = img.to(device)

                # 前向推理
                reconstructed = model(img, snr)

                # 将 PyTorch 张量转换为 NumPy 数组以便计算指标
                # .squeeze() 去掉 batch 维度, .cpu() 移回内存, .numpy() 转为 numpy
                # 维度互换：从 (C, H, W) 换回正常的图像 (H, W, C) 格式
                img_np = img.squeeze().cpu().numpy().transpose(1, 2, 0)
                recon_np = reconstructed.squeeze().cpu().numpy().transpose(1, 2, 0)

                # 计算指标
                # data_range=1.0 是因为图像被归一化到了 0~1 之间
                current_mse = mse(img_np, recon_np)
                current_psnr = psnr(img_np, recon_np, data_range=1.0)
                # 计算彩色图像的 SSIM 需要指定 channel_axis
                current_ssim = ssim(img_np, recon_np, data_range=1.0, channel_axis=-1)

                total_mse += current_mse
                total_psnr += current_psnr
                total_ssim += current_ssim

            # 计算当前 SNR 下的平均指标
            avg_mse = total_mse / len(dataloader)
            avg_psnr = total_psnr / len(dataloader)
            avg_ssim = total_ssim / len(dataloader)

            print(f"{snr:<10} | {avg_mse:<10.4f} | {avg_psnr:<10.2f} | {avg_ssim:<10.4f}")
    print("-" * 50)


if __name__ == '__main__':
    RANDOM_SEED = 42  # any random number
    def set_seed(seed):
        torch.manual_seed(seed)  # CPU
        torch.cuda.manual_seed(seed)  # GPU
        torch.cuda.manual_seed_all(seed)  # All GPU
        os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
        torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
        torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
    set_seed(RANDOM_SEED)

    evaluate_model()
