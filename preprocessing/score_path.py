"""
获取简单路径所对应的 relation 路径, 并计算该 relation 路径的得分
relation 路径的得分定义为, 使用该路径实例化得到的预测结果相比较答案的 HIT 得分
"""
import os
import glob
from copy import deepcopy
from tqdm import tqdm

from knowledge_graph.knowledge_graph import KnowledgeGraph
from knowledge_graph.knowledge_graph_cache import KnowledgeGraphCache
from knowledge_graph.knowledge_graph_freebase import KnowledgeGraphFreebase
from utils import load_jsonl, dump_jsonl
from config import cfg

def run_score_path():
    load_data_path = cfg.preprocessing["step2"]["load_data_path"]
    dump_data_path = cfg.preprocessing["step2"]["dump_data_path"]

    kg = KnowledgeGraphFreebase()

    data_list = load_jsonl(load_data_path)

    data_with_path_list = []
    for (item, paths) in tqdm(data_list):
        m = set()
        for path in paths:
            if isinstance(path, str):
                path = (path,)
            path = tuple(path)
            if path == ("type.object.type", "type.type.instance"):
                continue
            m.add(path)
        data_with_path_list.append((item, tuple(m)))

    def cal_path_val(topic_entity, path, answers):
        '''
        这个函数主要用于评估知识图谱中给定路径的预测准确性。它通过比较预测结果和正确答案集合来计算准确率。如果路径没有预测结果，则视为完全正确。
        这个函数可以用于评估知识图谱推理任务的效果。
        '''
        preds = kg.deduce_leaves_by_path(topic_entity, path)
        preds = set(preds)
        hit = preds & answers
        full = preds
        if not full:
            return 1
        '''
        计算并返回正确预测实体的数量除以所有预测实体的数量，得到预测的准确率。
        '''
        return len(hit) / len(full)

    m_list = []
    for item, p_strs in tqdm(data_with_path_list):
        answers = set(item['answers'])
        topic_entities = item['topic_entities']
        path_and_score_list = []
        for p_str in p_strs:
            path = p_str
            p_val_list = []
            for topic_entity in topic_entities:
                p_val_list.append(cal_path_val(topic_entity, path, answers))
            p_val = max(p_val_list)
            path_and_score_list.append(dict(path=path, score=p_val))
        m_item = deepcopy(item)
        m_item['path_and_score_list'] = path_and_score_list
        m_list.append(m_item)
    
    dump_jsonl(m_list, dump_data_path)
