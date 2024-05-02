import json
import os
import matplotlib.pyplot as plt

# 文件路径
file_path = 'MAE-baseline/MAE-baseline-withNormPixel/eval_finetune/output_dir'

log_path = os.path.join(file_path, "log.txt")

# 用于存储 epoch 和 test_acc1 的列表
epochs = []
test_acc1s = []

# 打开文件并逐行读取
with open(log_path, 'r') as file:
    for line in file:
        # 去除行尾的空白
        line = line.strip()
        # 解析 JSON 数据
        data = json.loads(line)
        # 提取 epoch 和 test_acc1
        epoch = data['epoch']
        test_acc1 = data['test_acc1']
        # 将数据添加到列表中
        epochs.append(epoch)
        test_acc1s.append(test_acc1)

# 绘制 epoch 和 test_acc1 的曲线
plt.plot(epochs, test_acc1s, marker='o', linestyle='-', color='b')
plt.xlabel('Epoch')
plt.ylabel('Test Acc1 (%)')
plt.title('Epoch vs. Test Acc1')
plt.grid(True)
# plt.show()

# 保存图片到与log.txt相同的目录下
output_filename = 'test_acc1_curve.png'
fig_path = os.path.join(file_path, output_filename)
plt.savefig(fig_path)