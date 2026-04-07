
import json

def convert_json_lines_to_array(input_file, output_file):
    json_array = []

    # 逐行读取 JSON Lines 文件
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # 去除行首尾的空白字符
            if line:  # 确保该行不为空
                try:
                    json_obj = json.loads(line)
                    json_array.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"解析错误: {e}")
                    print(f"错误行: {line[:200]}")  # 打印出错行的前 100 个字符
                    continue

    # 将列表转换为标准 JSON 格式并写入文件
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(json_array, file, ensure_ascii=False, indent=4)

# 示例用法
input_file = 'src/tmp/data/origin_nsm_data/webqsp/train_simple.json'  # 你的输入文件名
output_file = 'src/tmp/data/origin_nsm_data/webqsp/output.json'  # 转换后的输出文件名
convert_json_lines_to_array(input_file, output_file)
