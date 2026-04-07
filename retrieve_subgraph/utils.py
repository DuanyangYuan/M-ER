import json
from typing import List, Any

def load_jsonl(path: str):
    data_list = []
    index = 0
    with open(path, "r") as f:
        for line in f.readlines():
            index += 1
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"JSON解析错误，跳过第 {index} 行: {e}")



    return data_list

def dump_jsonl(data_list: List[Any], path: str):
    with open(path, "w") as f:
        for json_obj in data_list:
            f.write(json.dumps(json_obj) + "\n")
