import torch
import pandas as pd
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import math
import os

# ================= 配置区域 =================
# 这里的路径改为你合并后的完整模型路径
# (该目录下应该包含 config.json, model.safetensors, tokenizer.json 等文件)
MERGED_MODEL_PATH = "/private/home/wuhao/LLaMA-Factory/output/Qwen3-4B-Instruct-2507" 

CSV_FILE = "/private/home/wuhao/LLaMA-Factory/data/testSet-1000(2).csv"         # 你的测试输入文件
OUTPUT_FILE = "test_result_merged.csv" # 结果输出文件
NUM_GPUS = 8                       # 使用显卡数量
# ===========================================

def format_prompt(title):
    """
    Prompt 模板：必须与训练时保持一致
    """
    messages = [
        {"role": "system", "content": "You are an academic paper title reviewer. You must answer with only 'Y' or 'N'."},
        {"role": "user", "content": f"Determine whether the following paper title is valid:\n\n{title}"}
    ]
    return messages

def run_inference_on_chunk(rank, chunk_df, return_dict):
    """
    单个 GPU 的工作函数
    """
    try:
        # 显式指定当前进程使用的 GPU
        device = torch.device(f"cuda:{rank}")
        print(f"[GPU {rank}] 正在加载模型自: {MERGED_MODEL_PATH} ...")
        
        # 加载 Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
        
        # 加载合并后的模型 (直接加载到指定 device，不使用 device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_PATH,
            torch_dtype=torch.bfloat16, # A800 上推荐使用 bf16
            trust_remote_code=True,
            device_map=None 
        ).to(device)
        
        model.eval()
        
        results = []
        print(f"[GPU {rank}] 模型加载完毕，开始处理 {len(chunk_df)} 条数据")
        
        # 遍历数据进行推理
        for index, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), position=rank, desc=f"GPU {rank}"):
            input_title = row['title given by manchine']
            
            # 1. 构建 Prompt
            messages = format_prompt(input_title)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 2. 编码
            inputs = tokenizer([text], return_tensors="pt").to(device)
            
            # 3. 生成
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs.input_ids,
                    max_new_tokens=512,  # 根据需要调整
                    temperature=0.1,     # 测试建议低温度
                    do_sample=False      # 或者直接 greedy search
                )
            
            # 4. 解码
            # 只保留新生成的 token
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 保存结果
            results.append({
                "original_index": index,
                "model_prediction": response.strip()
            })
            
        return_dict[rank] = results
        print(f"[GPU {rank}] 任务完成！")
        
    except Exception as e:
        print(f"[GPU {rank}] 发生错误: {e}")
        return_dict[rank] = []

def main():
    # 设置多进程启动方式
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print("正在读取 CSV 数据...")
    if not os.path.exists(CSV_FILE):
        print(f"错误：找不到文件 {CSV_FILE}")
        return

# 尝试多种编码读取
    try:
        df = pd.read_csv(CSV_FILE, encoding='utf-8')
    except UnicodeDecodeError:
        print("警告: UTF-8 读取失败，正在尝试 GB18030 (兼容 GBK)...")
        try:
            df = pd.read_csv(CSV_FILE, encoding='gb18030')
        except UnicodeDecodeError:
            print("警告: GB18030 读取失败，正在尝试 Latin-1...")
            df = pd.read_csv(CSV_FILE, encoding='latin1')
    
    # 简单的列检查
    if 'title given by manchine' not in df.columns:
        print("错误：CSV 中缺少 'title given by manchine' 列")
        return

    # 数据分片
    total_len = len(df)
    chunk_size = math.ceil(total_len / NUM_GPUS)
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, total_len, chunk_size)]
    
    print(f"数据总量: {total_len}, 分配给 {NUM_GPUS} 张卡，每张卡约 {chunk_size} 条")

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    # 启动进程
    for rank in range(NUM_GPUS):
        if rank < len(chunks):
            p = mp.Process(target=run_inference_on_chunk, args=(rank, chunks[rank], return_dict))
            p.start()
            processes.append(p)

    # 等待结束
    for p in processes:
        p.join()

    print("正在合并所有 GPU 的结果...")
    
    # 合并结果
    all_results = []
    for rank in sorted(return_dict.keys()):
        all_results.extend(return_dict[rank])
    
    if not all_results:
        print("警告：没有生成任何结果，请检查报错信息。")
        return

    # 转换回 DataFrame 并还原顺序
    result_df = pd.DataFrame(all_results)
    result_df.set_index("original_index", inplace=True)
    
    # 将结果写入原表
    df['model_prediction'] = result_df['model_prediction']
    
    # 简单的准确率计算 (仅供参考)
    if 'Y/N' in df.columns:
        correct = 0
        valid_count = 0
        for idx, row in df.iterrows():
            try:
                pred = str(row['model_prediction']).strip().upper()
                truth = str(row['Y/N']).strip().upper()
                # 只要首字母一致就算对 (兼容 "N. xxx" 的情况)
                if pred and truth and pred[0] == truth[0]:
                    correct += 1
                valid_count += 1
            except:
                pass
        if valid_count > 0:
            print(f"估算准确率 (Y/N): {correct/valid_count:.2%}")

    # 保存
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"推理完成，结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()