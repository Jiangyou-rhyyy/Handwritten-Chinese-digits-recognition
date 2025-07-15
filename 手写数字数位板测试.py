import tkinter as tk
import PIL.ImageGrab as ImageGrab
import torch
import torchvision.transforms as transforms
import numpy as np
import PIL.ImageOps
from PIL import Image
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn as nn
import torchvision.transforms as transforms

SAVE_PATH = "your own save path"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 自动选择 GPU 或 CPU

# 定义 CNN 模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1),  # Conv1: 输入1通道 → 输出16通道
            torch.nn.ReLU(),  # ReLU 激活
            torch.nn.MaxPool2d(kernel_size=2,stride=2),  # MaxPool1: 尺寸减半

            torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),  # Conv2: 16→32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2),  # MaxPool2: 再次减半

            torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),  # Conv3: 32→64
            torch.nn.ReLU(),

            torch.nn.Flatten(),  # 展平成一维向量
            torch.nn.Linear(in_features= 7*7*64,out_features= 128),  # 全连接层1
            torch.nn.ReLU(),
            torch.nn.Linear(in_features= 128,out_features= 15)  # 全连接层2，输出15类 logits
        )
    
    def forward(self,input):
        output = self.model(input)  # 前向传播
        return output

model = Net().to(device)  # 将模型移动到 GPU 或 CPU
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))  # 加载已训练的权重
model.eval()  # 评估模式，不启用 dropout / BN 等

# 输入图片的预处理 pipeline
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整尺寸
    transforms.Grayscale(),  # 转灰度
    transforms.ToTensor(),  # 转为 Tensor，并归一化到 [0,1]
    transforms.Normalize((0.5,), (0.5,))  # 标准化到 [-1,1]
])

# 定义 Tkinter 应用程序
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("手写数字识别")
        self.geometry("300x400")
        self.resizable(False, False)

        # 画布用于手写输入
        self.canvas = tk.Canvas(self, bg="white", width=280, height=280)
        self.canvas.pack(pady=10)
        self.image = Image.new("L", (280, 280), color=255)  # 初始白色背景
        self.draw = ImageDraw.Draw(self.image)  # PIL ImageDraw 对象，用于同步绘图

        self.canvas.bind("<B1-Motion>", self.paint)  # 鼠标拖动时绘图

        btn_frame = tk.Frame(self)
        btn_frame.pack()

        tk.Button(btn_frame, text="识别", command=self.predict).pack(side="left", padx=10)  # 识别按钮
        tk.Button(btn_frame, text="清空", command=self.clear).pack(side="right", padx=10)  # 清空按钮

        self.result_label = tk.Label(self, text="", font=("Arial", 16))  # 显示识别结果
        self.result_label.pack(pady=10)

    def paint(self, event):
        # 画一个半径8的黑色圆点（模拟笔触）
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2, y2], fill=0)  # 同步到 PIL Image

    def clear(self):
        self.canvas.delete("all")  # 清空画布
        self.draw.rectangle([0, 0, 280, 280], fill=255)  # 清空 PIL Image
        self.result_label.config(text="")  # 清空结果标签

    def predict(self):
        img = self.image.copy()  # 获取当前手写图像
        img = ImageOps.invert(img)  # 反色：黑底白字
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(1, keepdim=True).item()

        # 映射字典
        mapping = {
            0: '零', 1: '一', 2: '二', 3: '三', 4: '四',
            5: '五', 6: '六', 7: '七', 8: '八', 9: '九',
            10: '十', 11: '百', 12: '千', 13: '万', 14: '亿'
        }

        result = mapping.get(pred, str(pred))  # 映射，如果不存在则直接显示数字
        self.result_label.config(text=f"识别结果: {result}")  # 显示结果


if __name__ == "__main__":
    app = App()
    app.mainloop()  # 启动 GUI 主循环
