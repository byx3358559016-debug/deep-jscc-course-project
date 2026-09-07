
### 2. 训练代码 (`train.py`)
"""
这段代码包含了数据集加载、U - Net
简化结构的定义、AWGN
信道模拟以及训练循环。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random


# ==========================================
# 1. 数据集定义 (加载本地 Kodak 数据集)
# ==========================================
class KodakDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        初始化数据集
        :param img_dir: 图片所在的文件夹路径 (例如 './kodak')
        :param transform: 图像预处理操作（如缩放、转为张量等）
        """
        self.img_dir = img_dir
        self.transform = transform
        # 获取文件夹下所有的 png 图片文件名
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def __len__(self):
        # 返回数据集包含的图片数量
        return len(self.img_names)

    def __getitem__(self, idx):
        # 根据索引读取对应的图片
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')  # 确保是RGB三通道彩色图

        if self.transform:
            image = self.transform(image)

        return image


# ==========================================
# 2. 语义通信网络模型定义 (简化版 U-Net 架构)
# ==========================================
class SemanticCommSystem(nn.Module):
    def __init__(self):
        super(SemanticCommSystem, self).__init__()

        # 发送端 (Transmitter / Encoder) - 负责提取语义特征并压缩
        self.encoder = nn.Sequential(
            # 输入通道3(RGB), 输出通道32, 卷积核大小3x3, 步长2(会使分辨率减半), 边缘填充1
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # 激活函数，加入非线性
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1)  # 最终提取出的语义特征(潜变量) # 16-->4,8,32
        )

        # 接收端 (Receiver / Decoder) - 负责从带噪特征中恢复原图
        self.decoder = nn.Sequential(
            # 转置卷积用于上采样(放大图片分辨率)
            nn.ConvTranspose2d(16, 64, kernel_size=3, stride=1, padding=1), # 16-->4,8,32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # 将输出限制在 0~1 之间，符合图像像素值的范围
        )

    def channel_forward(self, x, snr_db):
        """
        模拟无线信道 (AWGN信道)
        x: 发送端输出的特征矩阵
        snr_db: 信噪比 (dB)
        """
        # 如果是干净信道或信噪比极高，可以直接返回
        if snr_db is None:
            return x

        # 计算当前信号的平均功率
        signal_power = torch.mean(x ** 2)
        # 将 dB 单位的信噪比转换为线性值
        snr_linear = 10 ** (snr_db / 10.0)
        # 根据公式计算噪声功率
        noise_power = signal_power / snr_linear
        # 生成与信号维度相同的高斯白噪声
        noise = torch.randn_like(x) * torch.sqrt(noise_power)

        # 接收端接收到的信号 = 原始信号 + 噪声
        return x + noise

    def forward(self, x, snr_db):
        # 1. 编码提取特征 (发送端处理)
        encoded = self.encoder(x)
        # 2. 经过无线信道 (加噪声)
        received = self.channel_forward(encoded, snr_db)
        # 3. 解码恢复图像 (接收端处理)
        decoded = self.decoder(received)
        return decoded


# ==========================================
# 3. 训练主流程
# ==========================================
def main():
    # 检查是否有N卡GPU可用，没有则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    # 超参数设置 (可以通过修改这里来改变实验设置)
    BATCH_SIZE = 4
    EPOCHS = 200  # 训练轮数
    LEARNING_RATE = 1e-3
    RESOLUTION = 64  # 可选 32, 64, 128
    # 训练时让模型随机经历各种信噪比，以提高鲁棒性
    TRAIN_SNRS = [-10, -5, 0, 5, 10]

    # 数据预处理流水线
    transform = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),  # 统一缩放分辨率
        transforms.ToTensor()  # 将图片转换为 PyTorch 的张量，并将像素值归一化到 [0,1]
    ])

    # 加载数据集
    dataset = KodakDataset(img_dir='./kodak', transform=transform)
    # DataLoader 用于自动分批次(batch)打乱并提供数据
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 实例化模型并将其移动到 GPU(如果可用)
    model = SemanticCommSystem().to(device)

    # 定义损失函数：均方误差 (MSE)，计算恢复图像与原图的差异
    criterion = nn.MSELoss()
    # 定义优化器：Adam，用于自动更新网络权重
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("开始训练...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_imgs in dataloader:
            batch_imgs = batch_imgs.to(device)

            # 梯度清零，防止累加
            optimizer.zero_grad()

            # 随机选择一个 SNR 环境进行这一次的模拟
            current_snr = random.choice(TRAIN_SNRS)

            # 固定SNR=5
            # current_snr = 5

            # 前向传播：将图像输入网络，获取重建结果
            reconstructed = model(batch_imgs, current_snr)

            # 计算损失：对比原图和重建图
            loss = criterion(reconstructed, batch_imgs)

            # 反向传播：计算网络中每个参数的梯度
            loss.backward()

            # 优化更新：根据梯度调整网络权重
            optimizer.step()

            total_loss += loss.item()

        # 打印当前 Epoch 的平均损失
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    # 保存训练好的模型权重
    torch.save(model.state_dict(), 'semantic_model.pth')
    print("训练完成！模型已保存为 semantic_model.pth")


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

    main()