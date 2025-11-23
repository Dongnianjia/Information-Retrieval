import chardet
import os

# 你的两个CSV文件路径（根据实际位置修改，确保和nb-classifier中用的路径一致）
file_paths = [
    r"C:\Users\28923\Desktop\Citseer\positive_titles.csv",
    r"C:\Users\28923\Desktop\Citseer\negative_titles.csv"
]

# 逐个检测文件编码
for file_path in file_paths:
    # 读取文件字节流，检测编码
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # 读取前10000字节（足够检测）
        result = chardet.detect(raw_data)
    
    # 输出结果
    print(f"文件：{os.path.basename(file_path)}")
    print(f"检测到的编码：{result['encoding']}")
    print(f"置信度：{result['confidence']:.2f}\n")