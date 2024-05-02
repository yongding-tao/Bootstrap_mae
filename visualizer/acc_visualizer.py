import json
import os
import matplotlib.pyplot as plt

class AccVisualizer:
    def __init__(self):
        # 初始化一个字典，用于存储不同日志文件的数据
        self.data = {}
    
    def read_log_file(self, file_path, label):
        """读取一个日志文件，并存储其数据在字典中。
        
        参数:
            file_path: 日志文件的路径。
            label: 用于标记此日志文件的标签。
        """
        epochs = []
        test_acc1s = []
        
        # 打开日志文件并逐行读取
        with open(file_path, 'r') as file:
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
        
        # 存储数据在字典中
        self.data[label] = (epochs, test_acc1s)
    
    def plot_all_logs(self, output_filename=None):
        """绘制所有日志文件的数据在同一张图上。
        
        参数:
            output_filename: 可选参数，用于保存图像的文件名。
        """
        # 绘制所有读取的日志数据
        for label, (epochs, test_acc1s) in self.data.items():
            # plt.plot(epochs, test_acc1s, marker='o', linestyle='-', label=label)
            plt.plot(epochs, test_acc1s, linestyle='-', label=label)
        
        # 设置图形标签和标题
        plt.xlabel('Epoch')
        plt.ylabel('Test Acc1 (%)')
        plt.title('Comparison of Test Acc1')
        plt.legend()  # 显示图例
        plt.grid(True)
        
        # 如果指定了输出文件名，则保存图片
        if output_filename:
            plt.savefig(output_filename)
            print(f"Plot saved as {output_filename}")
        
        # 显示图像
        plt.show()

# 示例使用
# 请确保提供有效的文件路径和标签
file_paths = [
    ('../MAE-baseline/MAE-baseline-withNormPixel/eval_finetune/output_dir/log.txt', 'MAE baseline'),
    ('../Bootstrap_MAE/20240502-152734/eval_finetune/output_dir/log.txt', 'BMAE k=5 batch_size=256'),
    ('../Bootstrap_MAE/20240502-153005/eval_finetune/output_dir/log.txt', 'BMAE k=4 batch_size=128')
    # 其他文件路径和标签...
]

# 创建一个AccVisualizer实例
acc_visualizer = AccVisualizer()

# 读取多个日志文件
for file_path, label in file_paths:
    acc_visualizer.read_log_file(file_path, label)

# 绘制并显示所有日志文件的数据在同一张图上
acc_visualizer.plot_all_logs(output_filename='comparison_plot.png')