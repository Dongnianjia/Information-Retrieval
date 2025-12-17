import copy
import json
import os
import re
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

# ==========================================
# [新增] 1. 定义轻量级代理模型结构
# ==========================================
class TokenScorer(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128): # hidden 变大一点
        super(TokenScorer, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # [新增] 加入 Dropout，防止死记硬背
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # [新增]
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x).squeeze(-1)

# ==========================================
# [新增] 2. 初始化模型 (CodeBERT Embedding + Proxy)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading CodeBERT and Proxy Model...")

# 指向你之前手动下载好配置文件的目录
tokenizer = RobertaTokenizer.from_pretrained("./codebert-base")
bert_model = RobertaModel.from_pretrained("./codebert-base").to(device)
bert_model.eval() # 冻结

# 加载代理模型
proxy_model = TokenScorer().to(device)
# 注意：请确保目录下有训练好的 proxy_model.bin
if os.path.exists('proxy_model.bin'):
    proxy_model.load_state_dict(torch.load('proxy_model.bin', map_location=device))
    print("Proxy model loaded successfully.")
else:
    print("Warning: 'proxy_model.bin' not found. Using random weights (Result will be random).")
proxy_model.eval()

# [新增] 动态计算分数的辅助函数
def get_dynamic_attention(text):
    """输入代码字符串，经过Proxy模型，返回平均重要性分数"""
    if not text.strip():
        return 0.0
    try:
        # 截断长度防止OOM，只取 embedding
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            embeddings = bert_model.embeddings(inputs['input_ids'])
            scores = proxy_model(embeddings)
        # 返回该语句所有Token分数的平均值
        return scores.mean().item()
    except Exception as e:
        print(f"Error scoring text: {e}")
        return 0.001

# ==========================================
# 以下为原有逻辑保持不变
# ==========================================

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False

def assimilate_code_string_and_integer(code, string_mask=" string ", number_mask="10"):
    quotation_index_list = []
    for i in range(0, len(code)):
        if code[i] == "\"":
            quotation_index_list.append(i)
    for i in range(len(quotation_index_list) - 1, 0, -2):
        code = code[:quotation_index_list[i-1] + 1] + string_mask + code[quotation_index_list[i]:]

    tokens = code.split(" ")
    for i in range(0, len(tokens)):
        if is_number(tokens[i]):
            tokens[i] = number_mask
    code = " ".join(tokens)

    return code


def is_logger(statement):
    if statement.startswith('log ') or statement.startswith('Logger ') or ' Log ' in statement \
            or '. print ' in statement or ' . println ' in statement or 'LOG ' in statement \
            or statement.startswith('logger ') or statement.startswith('debug .'):
        return True
    return False

def is_getter(statement):
    if '. get ' in statement:
        return True
    return False

def is_setter(statement):
    if '. set' in statement and ')' in statement:
        return True
    return False

def is_if_statement(statement):
    if statement.startswith('if (') or statement.startswith('else if ') or statement.startswith('else '):
        return True
    return False

def is_while_statement(statement):
    if statement.startswith('while'):
        return True
    return False

def is_synchronized_statement(statement):
    if statement.startswith('synchronized '):
        return True
    return False

def is_for_statement(statement):
    if statement.startswith('for ('):
        return True
    return False

def is_throw_statement(statement):
    if statement.startswith('throw'):
        return True
    return False

def is_method_declaration_statement(statement):
    if statement.startswith('public ') or statement.startswith('protected ') or statement.startswith('private ') \
            or statement.startswith('@ Over ride') or statement.startswith('@ Bench mark') \
            or statement.startswith('@ Gener ated ') or statement.startswith('@ Test') \
            or statement.endswith(') {') or ("throws " in statement and 'Exception' in statement):
        return True
    return False

def is_switch_statement(statement):
    if statement.startswith('switch'):
        return True
    return False

def is_return_statement(statement):
    if statement.startswith('return'):
        return True
    return False

def is_variable_declaration_statement(statement):
    if statement.startswith('String ') or statement.startswith('int ') or statement.startswith('float ') \
            or statement.startswith('boolean') or statement.startswith('long') or statement.startswith('List') \
            or statement.startswith('Array ') or 'new ' in statement or statement.startswith('Collection') \
            or statement.startswith('final ') or "= true" in statement or "= null" in statement \
            or statement.startswith('Object ') or "= \" string \"" in statement or statement.startswith('Map <') \
            or statement.startswith('Class <') \
            and '=' in statement:
        return True
    return False

def is_reassign_statement(statement):
    if ")" not in statement and "=" in statement or " = (" in statement:
        return True
    return False

def is_try_statement(statement):
    if statement.startswith('try'):
        return True
    return False

def is_catch_statement(statement):
    if statement.startswith('catch'):
        return True
    return False

def is_finally_statement(statement):
    if statement.startswith('finally'):
        return True
    return False

def is_break_statement(statement):
    if statement.startswith('break'):
        return True
    return False

def is_case_statement(statement):
    if statement.startswith('case'):
        return True
    return False

def is_continue_statement(statement):
    if statement.startswith('continue'):
        return True
    return False

def is_expression(statement):
    if " * = " in statement or "++ ;" in statement or "/\ = " in statement \
            or "+=" in statement or "-- ;" in statement or " / = " in statement:
        return True
    return False

def is_function_caller(statement):
    if statement.endswith(') ;') and '(' in statement:
        return True
    return False

def is_annotation(statement):
    if statement.startswith('//') or statement.endswith('< p >') or statement.startswith('*/') \
            or statement.startswith('/*'):
        return True
    return False

def camel_case_split(str):
    RE_WORDS = re.compile(r'''
        [A-Z]+(?=[A-Z][a-z]) |
        [A-Z]?[a-z]+ |
        [A-Z]+ |
        \d+ |
        [^\u4e00-\u9fa5^a-z^A-Z^0-9]+
        ''', re.VERBOSE)
    return RE_WORDS.findall(str)

def merge_statements(code):
    statements = []
    tokens = code.split(' ')
    if len(tokens) > 2 and tokens[0] == '@':
        if tokens[2] == '(':
            try:
                start = tokens.index(')')
                tokens = tokens[start+1:]
            except:
                pass
        else:
            tokens = tokens[2:]
    current_token = []
    for i in range(len(tokens)):
        token = tokens[i]
        token = camel_case_split(token)
        for t in token:
            current_token.append(t)
    tokens = current_token
    start = 0
    index = start
    in_brace = 0
    endline_keyword = [';', '{', '}']
    while index < len(tokens):
        current_token = tokens[index]
        if current_token in ['(']:
            in_brace += 1
        elif current_token in [')']:
            in_brace -= 1
        if current_token in endline_keyword and in_brace > 0:
            index += 1
            continue
        if current_token in endline_keyword:
            statements.append(tokens[start:index+1])
            start = index + 1
            index += 1
            continue
        index += 1
    if start < len(tokens):
        statements.append(tokens[start:])
    return statements

def merge_statements_from_tokens(tokens):
    tokenIndexList = []
    start = 1
    end = 1
    result = []
    is_for_statement = False
    for i in range(1, len(tokens)):
        if tokens[i] == '</s>' and (i == len(tokens) - 1 or tokens[i + 1] == '<s>'):
            break
        if tokens[i] == '</s>':
            start = i + 1
            continue
        if is_for_statement:
            if '{' in tokens[i]:
                is_for_statement = False
                end = i
                result.append(tokens[start:end + 1])
                tokenIndexList.append([start, end])
                start = end + 1
            continue
        try:
            if tokens[i] == 'Ġfor' and tokens[i + 1] == 'Ġ(':
                is_for_statement = True
                start = i
                continue
        except:
            pass
        try:
            if (tokens[i] == 'Ġ>' and tokens[i - 1] == 'p' and tokens[i - 2] == 'Ġ<') \
                    or ';' in tokens[i] or '{' in tokens[i] or '}' in tokens[i]:
                end = i
                result.append(tokens[start:end + 1])
                tokenIndexList.append([start, end])
                start = end + 1
        except:
            pass
    return result, tokenIndexList


def get_statement_classification(statement,statement_classification_map):
    # 此函数主要保留用于获取 category 标签，attention 值将不再使用
    if is_try_statement(statement):
        return 'try', statement_classification_map.get('try', 0)
    elif is_catch_statement(statement):
        return 'catch', statement_classification_map.get('catch', 0)
    elif is_finally_statement(statement):
        return 'finally', statement_classification_map.get('finally', 0)
    elif is_break_statement(statement):
        return 'break', statement_classification_map.get('break', 0)
    elif is_continue_statement(statement):
        return 'continue', statement_classification_map.get('continue', 0)
    elif is_return_statement(statement):
        return 'return', statement_classification_map.get('return', 0)
    elif is_throw_statement(statement):
        return 'throw', statement_classification_map.get('throw', 0)
    elif is_annotation(statement):
        return 'annotation', statement_classification_map.get('annotation', 0)
    elif is_while_statement(statement):
        return 'while', statement_classification_map.get('while', 0)
    elif is_for_statement(statement):
        return 'for', statement_classification_map.get('for', 0)
    elif is_if_statement(statement):
        return 'if', statement_classification_map.get('if', 0)
    elif is_switch_statement(statement):
        return 'switch', statement_classification_map.get('switch', 0)
    elif is_expression(statement):
        return 'expression', statement_classification_map.get('expression', 0)
    elif is_synchronized_statement(statement):
        return 'synchronized', statement_classification_map.get('syncronized', 0)
    elif is_case_statement(statement):
        return 'case', statement_classification_map.get('case', 0)
    elif is_method_declaration_statement(statement):
        return 'method', statement_classification_map.get('method', 0)
    elif is_variable_declaration_statement(statement):
        return 'variable', statement_classification_map.get('variable', 0)
    elif is_logger(statement):
        return 'logger', statement_classification_map.get('log', 0)
    elif is_setter(statement):
        return 'setter', statement_classification_map.get('setter', 0)
    elif is_getter(statement):
        return 'getter', statement_classification_map.get('getter', 0)
    elif is_function_caller(statement):
        return 'function', statement_classification_map.get('function', 0)
    return 'None', 0.0001

def delete_with_algorithm_of_dietcode(code,target_len ,strategy,task):
    result = ''
    reduction = Code_Reduction(code, strategy, target_len,task)
    result = reduction.prune()
    return result

# 这些静态 Map 现在只用于兜底或分类判断，分数计算已被动态代理取代
dietcode_statement_classification_map = {} 
cls_statement_classification_map = {}

dietcode_lowest_ranked_token = []
cls_lowest_ranked_token = []

def get_token_attention():
    # 保持原有逻辑，防止报错，但在动态模式下作用有限
    try:
        with open('../utils/low_rated_word_dietcode', 'r') as f:
            for token in f.readlines():
                dietcode_lowest_ranked_token.append(token.replace('\n', ''))
        with open('../utils/low_rated_word_cls', 'r') as f:
            for token in f.readlines():
                cls_lowest_ranked_token.append(token.replace('\n', ''))
    except:
        pass

get_token_attention()

class Code_Reduction():
    def __init__(self, code, sterategy, targetLength,task):
        self.code = code
        self.sterategy = sterategy
        self.targetLength = targetLength
        self.task = task
        self.result = []
        self.generate_statements()

    def generate_statements(self):
        statements = merge_statements(self.code)
        self.statements = []
        
        for statement in statements:
            statement_str = ' '.join(statement)
            
            # 1. 获取类别
            category, _ = get_statement_classification(statement_str, {})
            
            # 2. 动态计算 Attention
            attention = get_dynamic_attention(statement_str)
            
            # 1. 核心语义层 (Method, Variable, Function) -> 极高保护
            if category in ['method', 'variable', 'function']:
                attention += 20.0  # 核心语义，名词和动词，尽可能保留
            
            # 2. 逻辑结构层 (Control Flow) -> 高保护
            elif category in ['return', 'throw', 'if', 'for', 'while', 'switch', 'case', 'try', 'catch']:
                attention += 10.0  # 逻辑骨架，保留以维持代码结构
            
            current_statement = {
                'category': category, 
                'content': statement,
                'length': len(statement), 
                'attention': attention
            }
            self.statements.append(current_statement)

    def prune_lowest_ranked_token(self, statements, prune_num, lowest_ranked_token):
        # 保持不变，这是辅助的 Token 级删除
        result = []
        candidate = []
        for statement in statements:
            for token in statement:
                if token in lowest_ranked_token:
                    attention_pos = lowest_ranked_token.index(token)
                    if len(candidate) <= prune_num:
                        candidate.append(attention_pos)
                    elif attention_pos < max(candidate) and attention_pos not in candidate:
                        candidate.remove(max(candidate))
                        candidate.append(attention_pos)
        
        pruned_num = 0
        need_check = True
        if candidate:
            candidate = [lowest_ranked_token[x] for x in candidate]
        else:
            need_check = False
            
        for statement in statements:
            if not need_check:
                result.append(statement)
                continue
            current_statement = []
            for token in statement:
                if token in candidate and need_check:
                    pruned_num += 1
                    if pruned_num >= prune_num:
                        need_check = False
                else:
                    current_statement.append(token)
            result.append(current_statement)
        return result

    def zero_one_backpack(self):
        max_length = 0
        for statement in self.statements:
            if statement['length'] > max_length:
                max_length = statement['length']
        # 防止 max_length 过大导致内存爆炸，做一个限制
        max_length = min(max_length + self.targetLength, 2048) 
        
        dp = [[{'attention': 0.0, 'statements': []}
               for i in range(max_length + 1)] for j in range(len(self.statements) + 1)]
        
        for i in range(1, len(self.statements) + 1):
            s_len = self.statements[i-1]['length']
            s_att = self.statements[i-1]['attention']
            
            for j in range(1, max_length + 1):
                # 不选当前语句
                current_map = {'attention': dp[i-1][j]['attention'],
                               'statements': copy.deepcopy(dp[i-1][j]['statements'])}
                dp[i][j] = current_map
                
                # 选当前语句
                if j >= s_len:
                    prev_att = dp[i-1][j-s_len]['attention']
                    if current_map['attention'] < prev_att + s_att:
                        dp[i][j]['attention'] = prev_att + s_att
                        dp[i][j]['statements'] = copy.deepcopy(dp[i-1][j-s_len]['statements'])
                        dp[i][j]['statements'].append(i-1)
                        
        # 寻找最接近 targetLength 且 attention 最大的解
        best_att = -1
        best_statements = []
        # 从 targetLength 开始向后搜索，找到收益最大的组合
        for j in range(self.targetLength, max_length + 1):
             if dp[-1][j]['attention'] > best_att:
                 best_att = dp[-1][j]['attention']
                 best_statements = dp[-1][j]['statements']
                 
        return best_att, best_statements

    def prune(self, **kwargs):
        total_attention, chosen_statements = self.zero_one_backpack()
        current_length = 0
        result = []
        for statement_index in chosen_statements:
            current_length += self.statements[statement_index]['length']
            result.append(self.statements[statement_index]['content'])
        
        pruned_token_num = current_length - self.targetLength
        
        # 如果背包算法选多了，进行二次 Token 级微调
        if pruned_token_num > 0:
            if self.sterategy == 'dietcode':
                result = self.prune_lowest_ranked_token(result, pruned_token_num, dietcode_lowest_ranked_token)
            elif self.sterategy == 'leancode_d':
                result = self.prune_lowest_ranked_token(result, pruned_token_num, cls_lowest_ranked_token)
            else:
                 # 默认处理
                 result = self.prune_lowest_ranked_token(result, pruned_token_num, [])
                 
        result_=[]
        for x in result:
            for y in x:
                result_.append(y)
        return ' '.join(result_[:self.targetLength])

def caculate_tokens(code):
    tokens = code.split(' ')
    if len(tokens) > 2 and tokens[0] == '@':
        if tokens[2] == '(':
            try:
                start = tokens.index(')')
                tokens = tokens[start+1:]
            except:
                pass
        else:
            tokens = tokens[2:]
    current_token = []
    for i in range(len(tokens)):
        token = tokens[i]
        token = camel_case_split(token)
        for t in token:
            current_token.append(t)
    tokens = current_token
    return len(tokens)

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string

if __name__ == '__main__':
    # 请修改为你的实际路径
    Dir = r'D:\homework\大三上\信息检索\作业3\LeanCode-master' 
    
    # 示例：仅生成 leancode_d (现在是 dynamic) 的数据
    strategies = ['leancode_d'] 
    
    for st in strategies:
        ratio = 0.9 # 剪枝比例
        output_dir = os.path.join(Dir, 'data', 'codesearch', st, '10') # 50% retention
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Processing strategy: {st}, Output: {output_dir}")
        
        # 输入文件路径，请根据实际情况修改
        input_file = os.path.join(Dir, 'data', 'codesearch', 'java_test_0_fli.jsonl')
        # 如果文件不存在，尝试默认路径
        if not os.path.exists(input_file):
             # 假设你在 codesearch 目录下运行
             input_file = r'../../codesearch_data/java_test_0.jsonl'
             
        if not os.path.exists(input_file):
            print(f"Error: Input file not found at {input_file}")
            continue

        with open(input_file, 'r', encoding='utf-8') as r, open(os.path.join(output_dir, 'test.txt'), 'w', encoding='utf-8') as w:
            lines = r.readlines()
            # 为了演示，只跑前100条，正式跑请去掉切片
            for line in tqdm(lines): 
                try:
                    line_dic = json.loads(line)
                    # 处理 Code
                    code_tokens = line_dic.get('code_tokens', [])
                    if not code_tokens:
                         code_tokens = line_dic.get('code', '').split()
                    
                    code = ' '.join([format_str(token) for token in code_tokens])
                    code = assimilate_code_string_and_integer(code)
                    
                    # 计算目标长度
                    original_len = caculate_tokens(code)
                    if original_len == 0: continue
                    target_len = int(original_len * ratio)
                    
                    # 执行剪枝 (核心调用)
                    pruned_code = delete_with_algorithm_of_dietcode(code, target_len, st, 'codesearch')
                    
                    # 处理 Docstring
                    doc_tokens = line_dic.get('docstring_tokens', [])
                    if not doc_tokens:
                        doc = line_dic.get('docstring', '').replace('\n', ' ')
                    else:
                        doc = ' '.join(doc_tokens)
                        
                    url = line_dic.get('url', '')
                    
                    # 写入 test.txt 格式: Code<CODESPLIT>Doc<CODESPLIT>URL
                    # 注意：原脚本可能是 code在前，但LeanCode标准格式通常是 Label... 
                    # 这里保持你提供的脚本输出格式: code <CODESPLIT> doc <CODESPLIT> url
                    # 如果 run_classifier 报错，可能需要调整顺序
                    new_line = f"{pruned_code}<CODESPLIT>{doc}<CODESPLIT>{url}\n"
                    w.write(new_line)
                except Exception as e:
                    print(f"Error processing line: {e}")
                    continue