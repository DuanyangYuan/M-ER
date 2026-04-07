# import json
#
#
# def load_json(file_path):
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             data = json.load(file)
#         return data
#     except json.JSONDecodeError as e:
#         print(f"JSONDecodeError: {e}")
#         print(f"错误位置：行 {e.lineno}, 列 {e.colno}, 字符 {e.pos}")
#
#         # 显示 JSON 文件中出错位置附近的内容
#         with open(file_path, 'r', encoding='utf-8') as file:
#             content = file.read()
#             error_snippet = content[e.pos - 50:e.pos + 50]
#             print("出错片段：", error_snippet)
#         return None
#
#
# json_data = load_json('/home/horanchen/ydy/study/code/SubgraphRetrievalKBQA-main/src/tmp/data/origin_nsm_data/webqsp/train_simple.json')
#
# print("====================")
# with open('/home/horanchen/ydy/study/code/SubgraphRetrievalKBQA-main/src/tmp/data/origin_nsm_data/webqsp/train_simple.json', 'r', encoding='utf-8') as file:
#     line_number = 0
#     for line in file:
#         line_number += 1
#         try:
#             json.loads(line)
#         except json.JSONDecodeError as e:
#             print(f"行 {line_number} 出现错误: {e}")
#             break
#
# print("====================")
# import json
#
# def load_multiple_json(file_path):
#     json_objects = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line_number, line in enumerate(file, start=1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 json_object = json.loads(line)
#                 json_objects.append(json_object)
#             except json.JSONDecodeError as e:
#                 print(f"行 {line_number} 出现错误: {e}")
#                 print("出错片段：", line[:100])  # 打印出错行的前100个字符
#                 break
#     return json_objects
#
# json_data = load_multiple_json('/home/horanchen/ydy/study/code/SubgraphRetrievalKBQA-main/src/tmp/data/origin_nsm_data/webqsp/train_simple.json')




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
input_file = '/home/horanchen/ydy/study/code/SubgraphRetrievalKBQA-main/src/tmp/data/origin_nsm_data/webqsp/train_simple.json'  # 你的输入文件名
output_file = '/home/horanchen/ydy/study/code/SubgraphRetrievalKBQA-main/src/tmp/data/origin_nsm_data/webqsp/output.json'  # 转换后的输出文件名
convert_json_lines_to_array(input_file, output_file)
