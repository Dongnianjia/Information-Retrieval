import random
import os
from tqdm import tqdm

# --- 配置 ---
# 输入: 指向 prune_dietcode.py 生成的【剪枝后】文件
input_file = r"D:\homework\大三上\信息检索\作业3\LeanCode-master\data\codesearch\leancode_d\50\test.txt"
# 输出: 生成一个新的带负样本的文件
output_file = r"D:\homework\大三上\信息检索\作业3\codesearch_data\train_valid\java\test_mrr_pruned.txt"

NUM_SAMPLES = 1000 
BATCH_SIZE = 1000

def main():
    print(f"1. 读取数据: {input_file}")
    all_lines = []
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 -> {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    total_data = len(all_lines)
    print(f"读取到 {total_data} 行数据。")
    
    parsed_data = []
    all_codes = []
    
    # --- 核心修复：兼容 3列 和 5列 格式 ---
    for line in all_lines:
        parts = line.split('<CODESPLIT>')
        
        # 情况A: 标准 5 列格式 (Label, Url, Func, Doc, Code)
        if len(parts) >= 5:
            parsed_data.append(line)
            all_codes.append(parts[-1]) # Code 在最后
            
        # 情况B: 剪枝脚本生成的 3 列格式 (Code, Doc, Url)
        elif len(parts) == 3:
            code = parts[0]
            doc = parts[1]
            url = parts[2]
            # 强行补全成 5 列标准格式，方便 run_classifier 读取
            # 补上 Label=1, Func=unknown
            standard_line = f"1<CODESPLIT>{url}<CODESPLIT>unknown_func<CODESPLIT>{doc}<CODESPLIT>{code}"
            parsed_data.append(standard_line)
            all_codes.append(code)
            
    valid_count = len(parsed_data)
    print(f"有效数据: {valid_count} 条")
    
    if valid_count < NUM_SAMPLES:
        print(f"警告：有效数据少于采样数 {NUM_SAMPLES}，将使用全部数据。")
        selected_indices = range(valid_count)
    else:
        random.seed(42)
        selected_indices = random.sample(range(valid_count), NUM_SAMPLES)
    
    print(f"2. 生成负样本数据...")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx in tqdm(selected_indices):
            # 1. 写入正样本
            pos_line = parsed_data[idx]
            f_out.write(pos_line + '\n')
            
            # 解析正样本里的各个字段
            parts = pos_line.split('<CODESPLIT>')
            # 标准格式下: 0:Label, 1:Url, 2:Func, 3:Doc, 4:Code
            url_part = parts[1]
            func_part = parts[2]
            doc_part = parts[3]
            
            # 2. 生成 999 个负样本
            distractor_indices = random.sample(range(len(all_codes)), BATCH_SIZE - 1)
            
            for neg_idx in distractor_indices:
                if neg_idx == idx: # 避免抽到自己
                     neg_idx = (neg_idx + 1) % len(all_codes)
                
                neg_code = all_codes[neg_idx]
                
                # 构造负样本行 (Label=0)
                # 保持 Url/Func/Doc 不变，只把 Code 换成错误的
                neg_line = f"0<CODESPLIT>{url_part}<CODESPLIT>{func_part}<CODESPLIT>{doc_part}<CODESPLIT>{neg_code}"
                f_out.write(neg_line + '\n')

    print(f"完成！已生成: {output_file}")

if __name__ == "__main__":
    main()