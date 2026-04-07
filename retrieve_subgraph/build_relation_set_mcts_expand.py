'''构建 Local KB 中 relation 集合以及 entity 集合 '''
import os
import json
from tqdm import tqdm
from utils import load_jsonl
# from loguru import logger
from config import cfg

'''
这段代码的目的是构建一个本地知识库（Local KB）中的实体（entity）集合和关系（relation）集合，并将这些集合输出到文本文件中。
这段代码通过遍历训练集、测试集和开发集中的每个JSON对象，提取其中的实体和关系，并将这些实体和关系汇总到两个集合中。最后，它将这两个集合转储到两个文本文件中。
'''
def run():
    load_data_path = cfg.retrieve_subgraph["dump_data_folder"]

    train_dataset = load_jsonl(os.path.join(load_data_path, "train_simple_mcts_expand_llmsim.json"))
    test_dataset = load_jsonl(os.path.join(load_data_path, "test_simple_mcts_expand_llmsim.json"))
    dev_dataset = load_jsonl(os.path.join(load_data_path, "dev_simple_mcts_expand_llmsim.json"))

    entity_set = set()
    relation_set = set()
    out_entity_set_filename = os.path.join(load_data_path, 'entities_mcts_expand_llmsim.txt')
    out_relation_set_filename = os.path.join(load_data_path, 'relations_mcts_expand_llmsim.txt')

    for dataset in [train_dataset, test_dataset, dev_dataset]:
        for json_obj in tqdm(dataset):
            answers = {ans_json_obj["kb_id"]
                    for ans_json_obj in json_obj["answers"]}
            # |: 这是集合的并集操作符。它将两个集合进行合并，返回一个新的集合，其中包含两个集合中所有的唯一元素。
            subgraph_entities = set(json_obj["subgraph"]["entities"]) | set(json_obj["entities"])
            subgraph_relations = {r for h, r, t in json_obj["subgraph"]["tuples"]}
            # 更新实体集合和关系集合。
            entity_set = entity_set | answers | subgraph_entities
            relation_set = relation_set | subgraph_relations

    def dump_list_to_txt(mylist, outname):
        with open(outname, 'w') as f:
            for item in mylist:
                print(item, file=f)

    entity_set = sorted(entity_set)
    relation_set = sorted(relation_set)

    dump_list_to_txt(entity_set, out_entity_set_filename)
    dump_list_to_txt(relation_set, out_relation_set_filename)
