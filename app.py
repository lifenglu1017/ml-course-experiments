import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 7 * 7, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc(x)
        return x

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

if os.path.exists('model.pth'):
    model.load_state_dict(torch.load('model.pth', map_location=device, weights_only=True))
    model.eval()
    print("✅ Model loaded successfully")
else:
    print("⚠️  model.pth not found, running in demo mode")

# 预测函数
def predict_digit(image):
    if image is None:
        return "请上传或绘制一张手写数字图片"
    
    if not os.path.exists('model.pth'):
        return "⚠️ 模型未加载（model.pth未找到）"
    
    try:
        # 预处理图片
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                img = Image.fromarray(image.astype('uint8'), 'RGB').convert('L')
            else:
                img = Image.fromarray(image.astype('uint8'), 'L')
        else:
            img = image.convert('L')
        
        img = img.resize((28, 28))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted = torch.argmax(probabilities).item()
            confidence = probabilities[predicted].item()
        
        # 生成结果
        result = f"🎯 预测结果: **{predicted}**\n\n"
        result += f"📊 置信度: {confidence*100:.2f}%\n\n"
        result += "---\n"
        result += "**Top-3 预测:**\n"
        
        top3 = torch.argsort(probabilities, descending=True)[:3]
        for i, idx in enumerate(top3):
            prob = probabilities[idx].item()
            result += f"{i+1}. 数字 {idx.item()}: {prob*100:.2f}%\n"
        
        return result
        
    except Exception as e:
        return f"❌ 预测错误: {str(e)}"

# 创建界面
with gr.Blocks(title="🔢 手写数字识别") as demo:
    gr.Markdown("# 🔢 手写数字识别系统")
    gr.Markdown("基于CNN的MNIST手写数字识别")
    
    with gr.Tab("📤 图片上传"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="上传手写数字图片", type="numpy")
                submit_btn = gr.Button("🔍 开始识别", variant="primary")
            with gr.Column():
                output = gr.Markdown(label="识别结果")
    
    with gr.Tab("✏️ 手写板"):
        with gr.Row():
            with gr.Column():
                canvas = gr.Image(label="✍️ 绘制数字", type="numpy", height=300)
                clear_btn = gr.Button("�️ 清空")
                sketch_btn = gr.Button("� 开始识别", variant="primary")
            with gr.Column():
                sketch_output = gr.Markdown(label="识别结果")
    
    gr.Markdown("---")
    gr.Markdown("### � 使用说明")
    gr.Markdown("""
    1. **图片上传**：上传一张手写数字图片（建议白底黑字）
    2. **手写板**：使用鼠标或触屏绘制数字
    3. 点击"开始识别"按钮获取结果
    """)

    # 绑定事件
    submit_btn.click(predict_digit, image_input, output)
    sketch_btn.click(predict_digit, canvas, sketch_output)
    clear_btn.click(lambda: (None, ""), [], [canvas, sketch_output])

if __name__ == "__main__":
    print("🚀 Starting Handwritten Digit Recognition System")
    print("📍 Access: http://localhost:7860")
    print("=" * 60)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
