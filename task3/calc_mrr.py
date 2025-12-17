import numpy as np

def main():
    # 刚才生成的预测结果文件
    result_file = r"./saved_models/java_baseline/mrr_results_pruned.tsv"
    
    print(f"正在读取预测结果: {result_file}")
    
    # 读取所有分数
    all_scores = []
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('<CODESPLIT>')
            if len(parts) >= 2:
                # 取正类的置信度分数
                all_scores.append(float(parts[-1]))
    
    # 转换成 numpy 数组
    all_scores = np.array(all_scores)
    
    # 每一组的大小 (1正 + 999负)
    GROUP_SIZE = 1000
    
    # 检查数据完整性
    if len(all_scores) % GROUP_SIZE != 0:
        print(f"警告：数据总数 {len(all_scores)} 不能被 {GROUP_SIZE} 整除，可能会有计算偏差。")
        # 截断多余的
        n_groups = len(all_scores) // GROUP_SIZE
        all_scores = all_scores[:n_groups * GROUP_SIZE]
    
    # 重塑数组: [Query数量, 1000]
    # 每一行的第 0 个元素 (index 0) 是正确答案的分数，其余是干扰项
    scores_matrix = all_scores.reshape(-1, GROUP_SIZE)
    
    print(f"成功加载 {scores_matrix.shape[0]} 组测试数据。")
    
    ranks = []
    for i in range(scores_matrix.shape[0]):
        group = scores_matrix[i]
        correct_score = group[0] # 第一个是正确答案
        
        # 计算有多少个分数 >= 正确答案的分数
        # (包括它自己，所以 Rank 最小是 1)
        rank = np.sum(group >= correct_score)
        ranks.append(rank)

    # 计算 MRR
    mrr = np.mean(1.0 / np.array(ranks))
    
    print("="*30)
    print(f"最终结果 (基于 {scores_matrix.shape[0]} 个Query的抽样评估):")
    print(f"MRR: {mrr:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()