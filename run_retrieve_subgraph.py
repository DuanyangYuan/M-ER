from retrieve_subgraph.retrieve_subgraph import run as run_retrieve_subgraph
from retrieve_subgraph.build_relation_set import run as run_build_relation_set

def run():
    # with open("/home/horanchen/ydy/study/code/SubgraphRetrievalKBQA-main/src/tmp/data/origin_nsm_data/webqsp/train_simple.json", "r") as f:
    #     lines = f.readlines()
    #     print(lines[0][177870:177910])  # 打印出出错附近的内容

    run_retrieve_subgraph()
    run_build_relation_set()

if __name__ == '__main__':
    run()
