import torch
import torchvision
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import pandas as pd
import os
import torchvision.transforms as transforms

DATASET_PATH = "your own data set path"
SAVE_PATH = "your own save path"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 选择设备

# 图像增强与预处理
transform = transforms.Compose([
    transforms.RandomAffine(  # 随机仿射变换：增加训练数据多样性
        degrees=15,
        scale=(0.9, 1.1),
        translate=(0.1, 0.1),
    ),
    transforms.Resize((28, 28)),  # 统一图像尺寸
    transforms.ToTensor(),  # 转换为 tensor 并归一化到 [0,1]
    transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化到 [-1,1] 范围
])

# 自定义 Dataset 类，读取 CSV 中的文件路径与标签
class ChineseMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, "data\\data")  # 图片存放路径
        self.labels_df = pd.read_csv(  # 读取标签 CSV 文件
            os.path.join(root_dir, "chinese_mnist.csv")
        )
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)  # 样本总数

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        suite_id = row['suite_id']
        sample_id = row['sample_id']
        code = int(row['code'])
        label = code - 1  # 标签从 0 开始（0~14）

        # 组装文件名并读取图像
        filename = f"input_{suite_id}_{sample_id}_{code}.jpg"
        img_path = os.path.join(self.data_dir, filename)
        image = Image.open(img_path).convert('L')  # 转为灰度图

        if self.transform:
            image = self.transform(image)  # 应用 transform

        return image, label

# 读取数据集路径
dataset_path = DATASET_PATH

# 加载数据集并划分训练 / 测试集（80% / 20%）
full_dataset = ChineseMNISTDataset(dataset_path, transform=transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
trainData, testData = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# DataLoader，用于 batch 加载数据
trainLoader = DataLoader(trainData, batch_size=64, shuffle=True)
testLoader = DataLoader(testData, batch_size=64, shuffle=False)

# CNN 模型定义
class Net(torch.nn.Module):
    def __init__(self, num_classes=15):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, 1, 1),  # Conv1: 1->16 通道
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # 池化1: 尺寸减半

            torch.nn.Conv2d(16, 32, 3, 1, 1),  # Conv2: 16->32 通道
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # 池化2: 再次减半

            torch.nn.Conv2d(32, 64, 3, 1, 1),  # Conv3: 32->64 通道
            torch.nn.ReLU(),

            torch.nn.Flatten(),  # 展平成 1D 向量
            torch.nn.Linear(7 * 7 * 64, 128),  # 全连接层1
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)  # 全连接层2，输出 15 类 logits
        )

    def forward(self, x):
        return self.model(x)  # 前向传播

model = Net(num_classes=15).to(device)  # 模型转到 GPU / CPU
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

epochs = 20
for epoch in range(epochs):
    model.train()  # 训练模式
    running_loss = 0.0
    for images, labels in tqdm(trainLoader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # 梯度清零
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算 loss
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1} - Loss: {running_loss/len(trainLoader):.4f}")  # 打印 epoch 平均 loss

# 测试阶段
model.eval()  # 评估模式
correct = 0
total = 0
with torch.no_grad():  # 测试不需要计算梯度
    for images, labels in testLoader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # 取最大 logits 的类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total  # 测试准确率
print(f"测试集准确率: {accuracy:.2f}%")

# 保存模型参数
torch.save(model.state_dict(), SAVE_PATH)
print("模型已保存到 pthmnist_cnn.pth")
