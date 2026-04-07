import networkx as nx
from collections import defaultdict
import numpy as np
import random
from tqdm import tqdm
import sys
from colorama import Fore
import os
import argparse
import math

# Argument parsing for configuration
parser = argparse.ArgumentParser(description='Path Finding for Relation Prediction')
parser.add_argument('--dataset', type=str, default='FB15k-237-subset', help='Dataset name')
parser.add_argument('--suffix', type=str, default='_1000', help='Suffix of the train file name')
parser.add_argument('--finding_mode', type=str, default='head', help='Specify if head, relation, or tail is fixed')
parser.add_argument('--training_mode', type=str, default='train', help='Specify if train, valid, test or interpret')
parser.add_argument('--data_dir', type=str, default=None, help='Directory to load data')
parser.add_argument('--output_dir', type=str, default=None, help='Directory to store output')
parser.add_argument('--train_dataset', type=str, default=None, help='Dataset for loading training graph')
parser.add_argument('--ranking_dataset', type=str, default=None, help='Dataset for ranking triplets')
parser.add_argument('--npaths_ranking', type=int, default=3, help='Number of paths for each triplet')
parser.add_argument('--support_threshold', type=float, default=5e-3, help='Path filtering threshold')
parser.add_argument('--support_type', type=int, default=2,
                    help='0: none, 1: relation path coverage, 2: relation path confidence')
parser.add_argument('--search_depth', type=int, default=5, help='Depth for path search')
parser.add_argument('--n_simulations', type=int, default=1, help='Number of MCTS simulations')
args = parser.parse_args()

# Initialize graph and data structures
G = nx.DiGraph()
relation_count = defaultdict(int)
train_triplets = []
ranking_triplets = []

# Define directories
data_dir = args.data_dir if args.data_dir else os.path.join("data/data/", args.dataset)
output_dir = args.output_dir or os.path.join('data/relation_prediction_path_data/', args.dataset,
                                             f"ranking_{args.finding_mode}{args.suffix}/")
ranking_dataset = args.ranking_dataset or os.path.join("data/relation_prediction_path_data/", args.dataset,
                                                       f"ranking_{args.finding_mode}{args.suffix}/ranking_{args.training_mode}.txt")
graph_file = "train_full.txt"

# Load graph and ranking triplets
with open(os.path.join(data_dir, graph_file), encoding='utf-8') as f:
    for line in f:
        h, r, t = line.split()
        train_triplets.append([h, r, t])
        G.add_edge(h, t, relation=r)
        G.add_edge(t, h, relation=f"{{{r}}}^-1")
        relation_count[r] += 1
        relation_count[f"{{{r}}}^-1"] += 1

with open(ranking_dataset, encoding='utf-8') as f:
    for line in f:
        h, r, t = line.split()
        ranking_triplets.append([h, r, t])


# Monte Carlo Tree Node
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


# Monte Carlo Tree Search (MCTS) algorithm
class MCTS:
    def __init__(self, graph, target, max_depth):
        self.graph = graph
        self.target = target
        self.max_depth = max_depth

    def select(self, node):
        """Select the best node to expand using UCT"""
        root = node
        path = []
        path.append(root.state)
        # 检查当前节点是否已经有孩子节点，如果没有则创建孩子节点
        if not node.children:
            if self.graph[node.state]:
                for v in self.graph[node.state]:
                    # 对每个邻居节点创建新的 Node 对象作为孩子节点
                    child_node = Node(state=v, parent=node)  # 假设 Node 类包含状态和父节点信息
                    node.children.append(child_node)  # 将新的孩子节点添加到当前节点的孩子列表中

                # 选择具有最高 UCT 值的孩子节点
        while node.children:
            node = max(node.children, key=lambda n: n.uct())
            path.append(node.state)
        # path = [root.state, node.state]
        return node, path

    def expand(self, node, path):
        """Expand the node by adding a child node with an unexplored action"""
        node.unexplored_actions = []
        unexplored_nodes = [v for v in self.graph[node.state] if v not in node.unexplored_actions]
        if unexplored_nodes:
            action = random.choice(unexplored_nodes)
            child_node = Node(state=action, parent=node)
            node.children.append(child_node)
            node.unexplored_actions.append(action)
            path.append(child_node.state)
            return child_node, path
        return node, path

    def simulate(self, root, node, path):
        """Simulate the game using BFS until we reach a terminal state (or max depth)"""
        # if node.state == self.target:
        #     # path = [root.state, node.state]
        #     return 1, path
        # 使用队列来实现广度优先搜索 (BFS)
        queue = []
        visited = defaultdict(int)
        parent_map = {}  # 用于记录每个节点的父节点，便于路径回溯

        queue.append([node.state, 0])
        visited[node.state] = 1
        visited[root.state] = 1
        # parent_map[node.state] = root.state
        # print(len(path))
        for i in range(len(path)):
            path[i] = Node(state=path[i])
            visited[path[i].state] = 1
            if i == 0:
                parent_map[path[i].state] = None
            else:
                parent_map[path[i].state] = path[i-1].state
        # if len(path) == 2:
        #     parent_map[node.state] = root.state  # 根节点没有父节点
        #     parent_map[root.state] = None
        # elif len(path) == 3:
        #     print(path)
        #     f_node = Node(state=path[1])
        #     visited[f_node.state] = 1
        #     parent_map[node.state] = f_node.state
        #     parent_map[f_node.state] = root.state
        #     parent_map[root.state] = None

        while len(queue) != 0:
            current_node, dpth = queue[0]  # 弹出队列中第一个节点 u 及其深度 dpth
            queue.pop(0)  # 移除队列中的第一个元素。

            if current_node not in self.graph.nodes or dpth >= self.max_depth:
                continue

            neighbors = list(self.graph[current_node])

            if not neighbors:
                return 0, []  # 死胡同，表示失败

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited[neighbor] = 1
                    parent_map[neighbor] = current_node  # 将每个邻居节点的父节点设为当前节点
                    queue.append([neighbor, dpth + 1])

                if neighbor == self.target:  # 找到目标，返回成功
                    # queue.pop(-1)
                    # 回溯路径
                    path = []
                    node_in_path = self.target
                    while node_in_path is not None:
                        path.append(node_in_path)  # 这里会陷入一个死循环，因为路径记录的不对，可能这个节点的父节点有很多，但是这里随便取了一个而不是取到对应的父节点
                        # if node_in_path == root.state:
                        #     break  # 根节点没有父节点，跳出循环
                        node_in_path = parent_map[node_in_path]  # 回溯父节点

                    path.reverse()  # 反转路径，得到从起始节点到目标节点的路径
                    # print(f"Found path: {path}")
                    return 1, path  # 返回路径
        return 0, []

    def backpropagate(self, node, result):
        """Backpropagate the result of the simulation"""
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

    def run_simulation(self, root):
        """Run one iteration of MCTS"""
        # Selection 这里选择的叶子节点也有问题，因为有好几个概率一样的，然后随机选择了一个，应该是对于概率一样的都应该进行遍历选择.
        leaf, path = self.select(root)
        if path[-1] == self.target:
            result = 1
            self.backpropagate(leaf, result)
            return result, path
        # Expansion
        if leaf.visits > 0:
            # print("=====1")
            leaf, path = self.expand(leaf, path)
            # print(path)

            if path[-1] == self.target:
                result = 1
                self.backpropagate(leaf, result)
                return result, path
        # Simulation
        result, path = self.simulate(root, leaf, path)
        # Backpropagation
        self.backpropagate(leaf, result)
        return result, path


def reset_wins(node):
    for child in node.children:
        child.wins = 0
        child.visits = 0
        reset_wins(child)  # 递归清零子节点的子节点


# Find paths using MCTS for head
def find_paths_mcts_head(G, triplets, n_simulations, max_depth):
    all_ranking_paths = []
    # ranking_paths.append([])

    pbar = tqdm(total=len(triplets), desc=args.training_mode, position=0, leave=True, file=sys.stdout,
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))

    for triplet in triplets:
        h, r, t = triplet
        root = Node(h)
        mcts = MCTS(G, t, max_depth)
        unique_paths = set()
        # Run multiple simulations
        ranking_paths = []
        for _ in range(n_simulations):
            # print(n_simulations)
            result, path = mcts.run_simulation(root)
            path_tuple = tuple(path)  # Convert path to tuple for set comparison
            if path:
                if path_tuple not in unique_paths:
                    unique_paths.add(path_tuple)  # Add the tuple to the set
                    ranking_paths.append(path)
        all_ranking_paths.append(ranking_paths)
        reset_wins(root)
        pbar.update(1)
    # Save paths to file
    save_paths_to_file(all_ranking_paths, G, "head")
    pbar.close()


# Save entity and relation paths to file
def save_paths_to_file(ranking_paths, G, mode):
    # Write entity paths
    with open(os.path.join(output_dir, f"entity_paths_{args.training_mode}_mcts.txt"), "w", encoding='utf-8') as f:
        for path_group in ranking_paths:
            f.write(str(len(path_group)) + "\n")
            for path in path_group:
                for entity in path:
                    f.write(entity + "\t")
                f.write("\n")

    with open(os.path.join(output_dir, f"relation_paths_{args.training_mode}_mcts.txt"), "w", encoding='utf-8') as f:
        for path_group in ranking_paths:
            f.write(str(len(path_group)) + "\n")
            for path in path_group:
                for i in range(len(path) - 1):
                    f.write(G[path[i]][path[i + 1]]['relation'] + "\t")
                f.write("\n")


# Main execution
if args.finding_mode == 'head':
    find_paths_mcts_head(G, ranking_triplets, args.n_simulations, args.search_depth)
