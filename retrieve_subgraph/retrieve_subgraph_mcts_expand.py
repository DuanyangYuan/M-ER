import argparse
import json
import os
import torch
import subprocess
from typing import Tuple, List, Any, Dict
from utils import dump_jsonl, load_jsonl
from tqdm import tqdm
from func_timeout import func_set_timeout, FunctionTimedOut
from transformers import AutoModel, AutoTokenizer
from models.model import *
# from loguru import logger
from prompt.prompts import *
from knowledge_graph.knowledge_graph import KnowledgeGraph
from knowledge_graph.knowledge_graph_cache import KnowledgeGraphCache
from knowledge_graph.knowledge_graph_freebase import KnowledgeGraphFreebase
from config import cfg
import networkx as nx
from collections import defaultdict
import random
import os
import math
import ast

END_REL = "END OF HOP"
TOP_K = 10
_min_score = 1e5
retrieval_model_ckpt = cfg.retriever_model_ckpt
device = torch.device('cuda:0')  #

print("[load model begin]")
kg = KnowledgeGraphCache()
tokenizer = AutoTokenizer.from_pretrained(retrieval_model_ckpt)
model = AutoModel.from_pretrained(retrieval_model_ckpt)
model = model.to(device)
print("[load model end]")


def path_to_subgraph(topic_entity: str, path: List[str]):
    return kg.deduce_subgraph_by_path(topic_entity, path)


def path_to_candidate_relations(topic_entity: str, path: List[str]) -> List[str]:
    new_relations = kg.deduce_leaves_relation_by_path(topic_entity, path)
    candidate_relations = [r for r in new_relations if r.split(".")[0] not in ["kg", "common"]]
    return list(candidate_relations)


@torch.no_grad()
def get_texts_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return embeddings

# gpt
def dfs_path_search(question: str, topic_entity: str, max_hop: int) -> List[Tuple[List[str], float]]:
    path_list = []
    path_score_list = [0]

    def dfs(current_path, current_score):
        if len(current_path) == max_hop:
            return [(current_path, current_score)]

        candidate_relations = path_to_candidate_relations(topic_entity, current_path)
        if not candidate_relations:
            return [(current_path, current_score)]

        results = []
        query_lined_list = ['#'.join([question] + current_path)]
        all_relation_list = list(set(candidate_relations))
        q_emb = get_texts_embeddings(query_lined_list).unsqueeze(1)  # [B, 1, D]

        target_emb = get_texts_embeddings(all_relation_list).unsqueeze(0)  # [1, L, D]
        sim_score = torch.cosine_similarity(q_emb, target_emb, dim=2) / 0.07  # [B, L]

        for relation in all_relation_list:
            new_path = current_path + [relation]
            j = all_relation_list.index(relation)
            new_score = float(sim_score[0, j]) + current_score  # 独立计算新路径的得分
            if new_score < 10:
                results.append((new_path, new_score))
            else:
                results.extend(dfs(new_path, new_score))  # 保证路径得分独立

        return results

    return dfs(path_list, 0)


def _reverse_graph(G: Dict[str, List[str]]):
    r_G = dict()
    for u in G:
        for v in G[u]:
            r_G.setdefault(v, []).append(u)
    return r_G


def bfs_graph(G: Dict[str, List[str]], root):
    visited = set()
    currentLevel = [root]
    while currentLevel:
        for v in currentLevel:
            visited.add(v)
        nextLevel = set()
        for v in currentLevel:
            for w in G.get(v, []):
                if w not in visited:
                    nextLevel.add(w)
        currentLevel = nextLevel
    return visited


def merge_graph(graph_l, root_l, graph_r, root_r):
    assert root_l != root_r
    all_nodes = set()
    common_nodes = set(graph_l) & set(graph_r)
    all_nodes |= common_nodes
    reverse_graph_l, reverse_graph_r = _reverse_graph(graph_l), _reverse_graph(graph_r)
    for node in common_nodes:
        ancestors_l = bfs_graph(reverse_graph_l, node)
        ancestors_r = bfs_graph(reverse_graph_r, node)
        descendants_l = bfs_graph(graph_l, node)
        descendants_r = bfs_graph(graph_r, node)
        all_nodes.update(ancestors_l)
        all_nodes.update(ancestors_r)
        all_nodes.update(descendants_l)
        all_nodes.update(descendants_r)
    return all_nodes


def filter_by_graph(nodes: List[str], triples: List[str], G: Dict[str, List[str]]):
    entities = set(G.keys())
    nodes = [e for e in nodes if e in entities]
    triples = [(h, r, t) for h, r, t in triples if h in entities and t in entities]
    return nodes, triples


def build_graph(nodes: List[str], triples: List[str]):
    G = {}
    for e in nodes:
        G[e] = []
    for h, _, t in triples:
        G.setdefault(h, []).append(t)
    return G


class Node:
    def __init__(self, state, parent=None):
        self.state = state  # current node state
        self.parent = parent  # parent node
        self.children = []  # child nodes
        self.visits = 0  # number of times visited
        self.wins = 0  # number of wins
        self.unexplored_actions = []  # actions that haven't been explored yet

    def uct(self, exploration_weight=1.41):
        """UCT score to select next node"""
        if self.visits == 0:
            # 有些几个都是这个值，但是返回之后取max只能取一个，可能是随机取得一个
            return float('inf')  # Prioritize unvisited nodes
        return (self.wins / self.visits) + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

def get_score(question, topic_entity, current_path, current_score,tar=None):

    candidate_relations = path_to_candidate_relations(topic_entity, current_path)
    if not candidate_relations:
        return [(current_path, current_score)]
    # # 滤除与current_path[-1]相同的关系
    # if current_path:
    #     candidate_relations = [rel for rel in candidate_relations if rel != current_path[-1]]
    # 滤除与current_path中所有节点相同的关系
    if current_path:
        # 将current_path转换为集合以提高查找效率
        current_path_set = set(current_path)
        # 使用列表推导式过滤掉存在于current_path中的候选关系
        candidate_relations = [rel for rel in candidate_relations if rel not in current_path_set]

    all_relation_list1 = list(set(candidate_relations))
    target_emb1 = get_texts_embeddings(all_relation_list1).unsqueeze(0)
    if tar is None:
        tar1 = target_emb1
    else:
        # tar1 = list(set(tar))
        tar1 = get_texts_embeddings(tar).unsqueeze(0)

    tar_score = torch.cosine_similarity(tar1, target_emb1, dim=2) / 0.07
    # if tar is None:
    #     tar_score = tar_score.diag()
    # else:
    #     # 如果 tar 不是 None，取每个 candidate 关系与 tar 的最大相似度
    #     tar_score, _ = tar_score.max(dim=2)  # [1, L]
    # 筛选出 tar_score > 0.5 的关系
    # 确保 tar_score 是 1 维张量 [L]
    if tar_score.dim() == 2:
        tar_score = tar_score.squeeze(0)  # [L]
    filtered_relations = []
    for rel, score in zip(candidate_relations, tar_score):  # tar_score 是 [1, L] 的张量
        if score > 0.5:
            filtered_relations.append(rel)

    if not filtered_relations:
        return [(current_path, current_score)]

    results = []
    query_lined_list = ['#'.join([question] + current_path)]
    all_relation_list = list(set(filtered_relations))
    q_emb = get_texts_embeddings(query_lined_list).unsqueeze(1)  # [B, 1, D]
    # print(all_relation_list)
    target_emb = get_texts_embeddings(all_relation_list).unsqueeze(0)  # [1, L, D]
    sim_score = torch.cosine_similarity(q_emb, target_emb, dim=2) / 0.07  # [B, L]
    for relation in all_relation_list:
        new_path = current_path + [relation]
        j = all_relation_list.index(relation)
        new_score = float(sim_score[0, j]) + current_score  # 独立计算新路径的得分
        # results.append((new_path, new_score))
        if new_score > 0.5:
            results.append((new_path, new_score))
        # else:
        #     results.extend([])  # 保证路径得分独立
    if not results:
        return [(current_path, current_score)]

    return results

# given prompt, generate proposal under instruction, unwrap is required
def get_proposal(prompt, method='llama', temperature=0.7, max_tokens=2048, seed=170, max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=1024):
    response = []
    cnt = 2
    while not response and cnt:
        response = local_inference_model(prompt, max_length=max_length, truncation=truncation, do_sample=do_sample,
                                            max_new_tokens=max_new_tokens, temperature=temperature)
        cnt -= 1
        # print('proposal: \n' + response)
    if not response:
        # print(f'获取<{method}>回复失败!\n')
        return []
    return response


class mcts_task(object):
    def __init__(self):
        super().__init__()
        self.propose_method = 'llama'
        self.temperature = 0.7
        self.max_tokens = 1024
        self.seed = 170
        self.max_length = 1024
        self.truncation = True
        self.do_sample = True
        self.max_new_tokens = 512
        self.use_reflection = 'common'

    @staticmethod
    def single_reflection_wrap(x: str, y: str = '') -> str:
        #     包装提示，用于生成单步反馈。
        #     参数：
        #         x：问题内容。
        #         y：已有步骤，默认为空。
        #         step：当前步数，默认为 0。
        #         lang：语言，默认为 'zh'。
        # print('\n', '==============================', 'single_reflection_wrap', '==============================',
        #       '\n')
        # print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤给出的意见:\n')
        prompt = single_reflection_prompt_zh + x + '\n已有节点:\n' + y + '\n输出:'
        return prompt

    def extract_reflection(self, p, step_n):
        #     从生成的内容中提取反馈。
        #     参数：
        #         p：生成的内容。
        #         step_n：当前步数。

        # 打印分隔符和内容。
        # print('-'*40 + 'In extract_reflection' + '-'*40)
        # print(p)
        # print('-'*40 + 'Out extract_reflection' + '-'*40)
        #     如果语言为中文，检查是否包含 '已解决' 或 <end>。
        #     如果问题已解决，返回 <end>。
        # if self.lang == 'zh':
        if '已解决' in p or '已经解决' in p or '<end>' in p:
            if step_n > 1:
                # print('此步问题已解决，停止下探。\n')
                return '<end>'
            else:
                return '<continue>'
        # 如果使用简化反馈，直接返回 <continue>。
        if self.use_reflection == 'simple':
            return '<continue>'
        # # 提取反馈内容，标准化格式。
        # if '意见:' not in p:
        #     print('输出格式有误！\n')
        #     return ''
        # revised_ = p.split('意见:')[1]
        # print(f'标准化后的意见:{p}\n')
        return p
        # 如果语言为英文，检查是否包含 'solved' 或 <end>。
        # 提取反馈内容，标准化格式
        # else:
        #     if ' solved' in p or '<end>' in p:
        #         print('标准化后的意见: <end>\n')
        #         return '<end>'
        #     else:
        #         if self.use_reflection == 'simple':
        #             return '<continue>'
        #         if 'Analysis:' not in p:
        #             print('输出格式有误！\n')
        #             return p
        #         revised_ = p.split('Analysis:')[1].strip()
        #         print(f'标准化后的意见:{revised_}\n')
        #         return revised_

    def get_reflection(self, x, y, step_n):
        # 构造提示，尝试生成反馈，最多尝试三次。
        reflection_prompt = self.single_reflection_wrap(x, y)

        cnt = 3
        response = []
        while not response and cnt:
            response = get_proposal(reflection_prompt, self.propose_method, self.temperature, self.max_tokens,
                                    self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, self.max_new_tokens)
            cnt -= 1
        # 如果生成失败，返回空字符串。
        if not response:
            # print('获得意见失败！\n')
            return ''

        # p = ''
        # for _ in response:
        #     p = p + _ + ' '
        # p = p.strip()
        # 提取并返回反馈内容。
        p = response
        return self.extract_reflection(p, step_n)

    @staticmethod
    def single_reflection_wrap_simple(x: str, y: str = '', step: int = 0) -> str:
        #     包装提示，用于生成简化的单步反馈。
        #     参数：
        #         x：问题内容。
        #         y：已有步骤，默认为空。
        #         step：当前步数，默认为 0。
        #         lang：语言，默认为 'zh'。
        # print('\n', '==============================', 'single_reflection_wrap_simple', '==============================',
        #       '\nstep: ', step)
        # print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤给出的意见:\n')
        # if lang == 'zh':
        if not y:
            y = '无\n'
        prompt = single_reflection_prompt_simple + x + '\n已有步骤:\n' + y + '\n输出:'  # simple style
        # else:
        #     if not y:
        #         y = 'None\n'
        #     prompt = single_reflection_prompt_simple_en.format(problem=x, steps=y)
        #     # print(prompt)
        return prompt

    def get_simple_reflection(self, x, y, step_n):
        # 如果是第一步，直接返回 <continue>。
        if step_n == 1:
            return '<continue>'
        # if self.lang == 'en':
        #     if 'answer is' in y:
        #         return '<end>'
        # 构造提示，尝试生成反馈，最多尝试三次。
        reflection_prompt = self.single_reflection_wrap_simple(x, y, step_n)
        cnt = 3
        response = []
        while not response and cnt:
            response = get_proposal(reflection_prompt, self.propose_method, self.temperature, self.max_tokens,
                                    self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, 128)
            cnt -= 1
        # 如果生成失败，返回 <end>。
        if not response:
            # print('获得意见失败！\n')
            return '<end>'
        #    提取并返回反馈内容。
        # print('-'*40 + 'In get_simple_reflection' + '-'*40)
        # print(response)
        # print('-'*40 + 'Out get_simple_reflection' + '-'*40)

        # p = ''
        # for _ in response:
        #     p = p + _ + ' '
        # p = p.strip()
        p = response

        return self.extract_reflection(p, step_n)

# Monte Carlo Tree Search (MCTS) algorithm
class MCTS(mcts_task):
    def __init__(self, graph, max_depth):
        super().__init__()
        self.graph = graph
        self.max_depth = max_depth
        self.use_reflection = 'common'

    def select(self, question, node, list_node):
        path_list = []

        """Select the best node to expand using UCT"""
        root = node
        path = []
        result_paths = []
        best_score = 0
        # 检查当前节点是否已经有孩子节点，如果没有则创建孩子节点
        if not node.children:
            candidate_relations = path_to_candidate_relations(node.state, path)
            if candidate_relations:
                for v in candidate_relations:
                    if not any(n.state == v for n in list_node):
                        # 对每个邻居节点创建新的 Node 对象作为孩子节点
                        child_node = Node(state=v, parent=node)  # 假设 Node 类包含状态和父节点信息
                        node.children.append(child_node)  # 将新的孩子节点添加到当前节点的孩子列表中
                        list_node.append(child_node)

        path.append(root.state)
        if node.children:
            while node.children:
                # node = max(node.children, key=lambda n: n.uct())
                all_relation_list = get_score(question, root.state, path[1:], best_score)
                result_paths = sorted(all_relation_list, key=lambda x: x[1], reverse=True)[
                               :1]
                # node = Node(result_paths[0])
                # path.append(node.state)

                if result_paths:  # 确保结果不为空
                    best_path, best_score = result_paths[0]  # 获取分数最高的路径和分数
                    if best_path:
                        # print("best_path")
                        # print(best_path)
                        # print("list_node")
                        # print(list_node)
                        existing_node = next((n for n in list_node if n.state == best_path[-1]), None)
                        if existing_node is not None:
                            # 如果存在，直接将该节点的state添加到path后面
                            path.append(existing_node.state)
                            node = existing_node  # 更新当前节点为已存在的节点
                        else:
                            # 如果不存在，创建新的Node对象
                            new_node = Node(state=best_path[-1], parent=node)
                            node = new_node  # 更新当前节点为新节点
                            path.append(best_path[-1])  # 将新节点的状态添加到路径中
                            list_node.append(new_node)  # 将新节点添加到list_node列表中

                else:
                    # print("5=======")
                    break  # 如果没有结果，退出循环
        else:
            # print("1====")
            result_paths = [(path, 0)]
            # print(result_paths)
        return node, path, result_paths

    def  get_re(self, result_paths, question):
        if not result_paths:
            if self.use_reflection == "common":
                result_paths_str = 'no'
                reflection = self.get_reflection(question, result_paths_str, len(result_paths))
            else:  # simple
                reflection = self.get_simple_reflection(question, result_paths, len(result_paths))
        else:
            if self.use_reflection == "common":
                result_paths_str = result_paths[0][0][0]
                reflection = self.get_reflection(question, result_paths_str, len(result_paths))
            else:  # simple
                reflection = self.get_simple_reflection(question, result_paths, len(result_paths))
        return reflection

    def expand(self, question, root, node, path, result_paths, tar):
        reflection = self.get_re(result_paths, question)
        # print(reflection)
        if reflection == '<end>':
            return node, path, result_paths[0][1]
        """Expand the node by adding a child node with an unexplored action"""
        node.unexplored_actions = []
        # unexplored_nodes = [v for v in self.graph[node.state] if v not in node.unexplored_actions]
        p = path[-1]
        all_relation_list = get_score(question, root.state, path[1:], result_paths[0][1],tar)
        result_paths1 = sorted(all_relation_list, key=lambda x: x[1], reverse=True)[
                       :10]
        # node = Node(result_paths[0])
        if len(result_paths1)>1:
            chosen_list = random.choice(result_paths1)
            # print(chosen_list)
            action = chosen_list[0][-1]
            score = chosen_list[1]
            if action != node.state:
                child_node = Node(state=action, parent=node)
                node.children.append(child_node)
                node.unexplored_actions.append(action)
                if child_node.state != path[-1]:
                    path.append(child_node.state)
                return child_node, path, score
        return node, path, result_paths[0][1]

    def backpropagate(self, node, result):
        """Backpropagate the result of the simulation"""
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

    def simulate(self, question, root,  node, path_node, result_paths_score, tar):
        # 使用队列来实现广度优先搜索 (BFS)
        queue = []
        visited = defaultdict(int)
        parent_map = {}  # 用于记录每个节点的父节点，便于路径回溯

        queue.append([node.state, 0])
        visited[node.state] = 1
        visited[root.state] = 1
        # parent_map[node.state] = root.state
        # print(len(path))
        result_path = path_node.copy()
        result_paths = result_paths_score.copy()
        # print("3====")
        # print(result_paths)
        for i in range(len(path_node)):
            path_node[i] = Node(state=path_node[i])
            visited[path_node[i].state] = 1
            if i == 0:
                parent_map[path_node[i].state] = None
            else:
                parent_map[path_node[i].state] = path_node[i-1].state

        while len(queue) != 0:
            current_node, dpth = queue[0]  # 弹出队列中第一个节点 u 及其深度 dpth
            queue.pop(0)  # 移除队列中的第一个元素。

            if dpth >= self.max_depth:
                break
            ner_re = [ner for ner in result_path[1:]]
            # print("2====")
            # print(result_paths)
            neighbors_list = get_score(question, root.state, ner_re, result_paths[0][1], tar)
            result_paths = sorted(neighbors_list, key=lambda x: x[1], reverse=True)[
                           :1]
            # if not neighbors_list:
            #     return 0, []  # 死胡同，表示失败
            # print("result_paths")
            # print(result_paths)
            a = [neighbor_i[0] for neighbor_i in result_paths]
            # print("a")
            # print(a)

            neighbors_list_for = [neighbor_i[0] for neighbor_i in result_paths][0]

            for_result_path = result_path[1:]
            if neighbors_list_for != for_result_path:
                neighbors_list_list = [neighbor_i[0][-1] for neighbor_i in result_paths]
                neighbors_list_score = [neighbor_i[1] for neighbor_i in result_paths]
                for i, neighbor in enumerate(neighbors_list_list):
                    if neighbor not in visited:
                        visited[neighbor] = 1
                        parent_map[neighbor] = ner_re[-1]  # 将每个邻居节点的父节点设为当前节点
                        queue.append([neighbor, dpth + 1])


                    if neighbors_list_score[i] < 0.5:  # 找到目标，返回成功
                        # queue.pop(-1)
                        # 回溯路径
                        path = []
                        node_in_path = result_path[-1]
                        while node_in_path is not None:
                            path.append(node_in_path)  # 这里会陷入一个死循环，因为路径记录的不对，可能这个节点的父节点有很多，但是这里随便取了一个而不是取到对应的父节点
                            # if node_in_path == root.state:
                            #     break  # 根节点没有父节点，跳出循环
                            node_in_path = parent_map[node_in_path]  # 回溯父节点

                        path.reverse()  # 反转路径，得到从起始节点到目标节点的路径
                        # print(f"Found path: {path}")
                        return 1, path, neighbors_list_score[i]  # 返回路径
                    else:
                        result_path.append(neighbor)
            else:
                # 回溯路径
                path = []
                node_in_path = result_path[-1]
                while node_in_path is not None:
                    path.append(node_in_path)  # 这里会陷入一个死循环，因为路径记录的不对，可能这个节点的父节点有很多，但是这里随便取了一个而不是取到对应的父节点
                    # if node_in_path == root.state:
                    #     break  # 根节点没有父节点，跳出循环
                    node_in_path = parent_map[node_in_path]  # 回溯父节点

                path.reverse()  # 反转路径，得到从起始节点到目标节点的路径
                # print(f"Found path: {path}")
                score = [neighbor_i[1] for neighbor_i in result_paths]
                return 1, path, score # 返回路径

        return 1, neighbors_list_for, neighbors_list_score

    def run_simulation(self, question, root, list_node, tar):
        # Selection 这里选择的叶子节点也有问题，因为有好几个概率一样的，然后随机选择了一个，应该是对于概率一样的都应该进行遍历选择.
        # print("1======")
        leaf, path, result_paths = self.select(question, root, list_node)
        # if path[-1] == self.target:
        #     result = 1
        #     self.backpropagate(leaf, result)
        #     return result, path

        # Expansion
        if leaf.visits > 0:
            # print("=====1")
            leaf, path, score = self.expand(question, root, leaf, path, result_paths,tar)
            # print(path)

            if score < 0.5:
                result = 1
                self.backpropagate(leaf, result)
                return result, path, score
        # Simulation
        # print("3======")
        # print(result_paths)
        if len(result_paths) != 0:
            result, path, score = self.simulate(question, root, leaf, path, result_paths, tar)
        # Backpropagation
        #     print("4======")
            self.backpropagate(leaf, result)
            return result, path, score
        else:
            return 0, path, 0
import re
def clean_reflection(reflection: str) -> str:
    """
    清理 reflection 字符串，确保实体名称和关系名称被正确包裹在引号中。
    """
    # 移除非法标点符号（但保留逗号和空格）
    reflection = re.sub(r"[?!,;:]", "", reflection)
    reflection = reflection.strip()
    reflection = reflection.replace("\n", "")  # 移除换行符

    # 使用正则表达式匹配实体和关系名称
    # 假设实体和关系名称由字母、数字、下划线组成
    pattern = re.compile(r"[\w]+")
    matches = pattern.findall(reflection)

    # 将匹配到的内容重新组合为一个合法的 Python 列表字符串
    cleaned_reflection = "[" + ", ".join(f'"{match}"' for match in matches) + "]"
    return cleaned_reflection

def reset_wins(node):
    for child in node.children:
        child.wins = 0
        child.visits = 0
        reset_wins(child)  # 递归清零子节点的子节点

def retrieve_subgraph(json_obj: Dict[str, Any], entities):
    question = json_obj["question"]
    if len(json_obj["entities"]) == 0:
        return

    answers = set([ans_obj["kb_id"] for ans_obj in json_obj["answers"]])

    paths = []
    graphs = []

    G = nx.DiGraph()
    max_depth = 2
    mcts = MCTS(G, max_depth)

    for entity_id in json_obj["entities"]:

        topic_entity = entities[entity_id]
        root = Node(topic_entity)
        unique_paths = set()
        ranking_paths = []
        list_node = []

        reflection = mcts.get_re(paths, question)
        reflection =  clean_reflection(reflection)
        reflection = ast.literal_eval(reflection)
        # print(reflection)
        if '<end>' in reflection:
            break
        for tar in reflection:
            for _ in range(3):
                # print(n_simulations)
                result, path, score = mcts.run_simulation(question, root, list_node, tar)
                path_tuple = tuple(path)  # Convert path to tuple for set comparison
                if path:
                    if path_tuple not in unique_paths:
                        unique_paths.add(path_tuple)  # Add the tuple to the set
                        ranking_paths.append((path,score))

        # path_score_list = dfs_path_search(question, topic_entity, 3)
        nodes = []
        triples = []

        min_score = 1e5
        for path, score in ranking_paths:
            partial_nodes, partial_triples = path_to_subgraph(topic_entity, path[1:])
            if len(partial_nodes) > 1000:
                continue
            paths.append((topic_entity, path))
            nodes.extend(partial_nodes)
            triples.extend(partial_triples)

            if len(answers & set(partial_nodes)) > 0:
                min_score = min(min_score, score[0])
            # if len(nodes) > 1000:
            #     break
        graphs.append((topic_entity, nodes, triples, build_graph(nodes, triples)))

    n = len(graphs)
    for i in range(1, n):
        g0 = graphs[0]
        gi = graphs[i]
        topic_entity = g0[0]
        nodes = merge_graph(g0[3], g0[0], gi[3], gi[0])
        triples = [(h, r, t) for h, r, t in list(set(g0[2]) | set(gi[2])) if h in nodes and t in nodes]
        graph = build_graph(nodes, triples)
        graphs[0] = (topic_entity, nodes, triples, graph)

    nodes = graphs[0][1]
    triples = graphs[0][2]

    global _min_score
    _min_score = min(_min_score, min_score)

    nodes = list(set(nodes))
    triples = list(set(triples))
    subgraph_entities = [e for e in nodes]
    subgraph_tuples = [(h, r, t) for h, r, t in triples]
    json_obj["paths"] = paths
    json_obj["entities"] = [entities[e] for e in json_obj["entities"]]
    json_obj["subgraph"] = {
        "tuples": subgraph_tuples,
        "entities": subgraph_entities
    }


def build_entities(load_data_path):
    entities = []
    with open(os.path.join(load_data_path, "entities.txt"), "r") as f:
        for line in f.readlines():
            entities.append(line.strip())
    return entities

import time
def run():
    load_data_folder = cfg.retrieve_subgraph["load_data_folder"]
    dump_data_folder = cfg.retrieve_subgraph["dump_data_folder"]

    if not os.path.exists(dump_data_folder):
        os.makedirs(dump_data_folder)

    subprocess.run(["cp", "-r", load_data_folder, os.path.dirname(dump_data_folder)])

    train_dataset = load_jsonl(os.path.join(load_data_folder, "train_simple.json"))
    test_dataset = load_jsonl(os.path.join(load_data_folder, "test_simple.json"))
    dev_dataset = load_jsonl(os.path.join(load_data_folder, "dev_simple.json"))

    entities = build_entities(load_data_folder)
    total_time = 0  # 用于存储总时间
    num_calls = 0  # 用于存储函数调用次数
    max_time = 0  # 用于存储最长的执行时间
    for json_obj in tqdm(train_dataset, desc="retrieve:train"):
        start_time = time.time()
        retrieve_subgraph(json_obj, entities)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算经过时间
        total_time += elapsed_time  # 累加到总时间
        num_calls += 1  # 增加函数调用次数
        max_time = max(max_time, elapsed_time)  # 更新最长执行时间
        # print(f"Time taken for retrieve_subgraph: {elapsed_time:.2f} seconds")
    average_time = total_time / num_calls if num_calls > 0 else 0
    print(f"Average time per retrieve_subgraph call: {average_time:.4f} seconds")
    print(f"Maximum time taken for a single retrieve_subgraph call: {max_time:.4f} seconds")  # 打印最长时间

    for json_obj in tqdm(test_dataset, desc="retrieve:test"):
        retrieve_subgraph(json_obj, entities)

    for json_obj in tqdm(dev_dataset, desc="retrieve:dev"):
        retrieve_subgraph(json_obj, entities)

    dump_jsonl(train_dataset, os.path.join(dump_data_folder, "train_simple_mcts_expand_llmsim.json"))
    dump_jsonl(test_dataset, os.path.join(dump_data_folder, "test_simple_mcts_expand_llmsim.json"))
    dump_jsonl(dev_dataset, os.path.join(dump_data_folder, "dev_simple_mcts_expand_llmsim.json"))

    print("min score:", _min_score)


if __name__ == "__main__":
    run()