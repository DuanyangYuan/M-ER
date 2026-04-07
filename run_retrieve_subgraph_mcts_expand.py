from retrieve_subgraph.retrieve_subgraph_mcts_expand import run as run_retrieve_subgraph_mcts_expand
from retrieve_subgraph.build_relation_set_mcts_expand import run as run_build_relation_set_mstc_expand


def run():
    run_retrieve_subgraph_mcts_expand()
    run_build_relation_set_mstc_expand()


if __name__ == '__main__':
    run()
