import json, os
from biqe.util import *


def process_bio():
    data_dir = './data'
    graph, feature_modules, node_maps = load_graph(data_dir, 128)
    print("Loading edge data..")
    test_queries = load_test_queries_by_formula(data_dir + "/test_edges.pkl")
    train_queries = load_queries_by_formula(data_dir + "/train_edges.pkl")
    val_queries = load_test_queries_by_formula(data_dir + "/val_edges.pkl")

    print("Loading query data..")
    for i in range(2, 4):
        train_queries.update(load_queries_by_formula(data_dir + "/train_queries_{:d}.pkl".format(i)))
        i_val_queries = load_test_queries_by_formula(data_dir + "/val_queries_{:d}.pkl".format(i))
        val_queries["one_neg"].update(i_val_queries["one_neg"])
        val_queries["full_neg"].update(i_val_queries["full_neg"])
        i_test_queries = load_test_queries_by_formula(data_dir + "/test_queries_{:d}.pkl".format(i))
        test_queries["one_neg"].update(i_test_queries["one_neg"])
        test_queries["full_neg"].update(i_test_queries["full_neg"])
    # bhushan
    write_queries(test_queries, 'test_one.json', is_train=False)
    write_queries(val_queries, 'dev_one.json', is_train=False)
    write_queries(train_queries, 'train.json', is_train=True)
    exit()


def gen_path(q):
    # note that target is at beg of array and anchors at end.
    path = []
    if q.formula.query_type in {'1-chain','2-chain','3-chain'}:
        assert len(q.anchor_nodes)==1
        path = ['[MASK]'] + ['-'.join(p) for p in q.formula.rels] + [str(q.anchor_nodes[0])]
        path = '-#-'.join(path)
    elif q.formula.query_type in {'2-inter','3-inter'}:
        path = []
        for count, s in enumerate(q.anchor_nodes):
            segment = ['[MASK]']
            rel = '-'.join(q.formula.rels[count])
            segment.append(rel)
            segment.append(str(s))
            path.append('-#-'.join(segment))
        path = '[SEP]'.join(path)
    elif q.formula.query_type =='3-chain_inter':
        path = []
        for count, s in enumerate(q.anchor_nodes):
            segment = ['[MASK]']
            segment.append('-'.join(q.formula.rels[0]))
            rel = '-'.join(q.formula.rels[-1][count])
            segment.append(rel)
            segment.append(str(s))
            path.append('-#-'.join(segment))
        path = '[SEP]'.join(path)
    elif q.formula.query_type == '3-inter_chain':
        path = []
        for count,s in enumerate(q.anchor_nodes):
            segment = ['[MASK]']
            if count<=0:
                rel = '-'.join(q.formula.rels[count])
            else:
                rel = '-'.join(q.formula.rels[count][0]) + '-#-' + '-'.join(q.formula.rels[count][1])
            segment.append(rel)
            segment.append(str(s))
            path.append('-#-'.join(segment))
        path = '[SEP]'.join(path)
    return path

def write_queries(data_queries, f_name, is_train=False):
    data = []
    data_dir = os.path.join("./data",f_name)
    data_queries = data_queries if is_train else data_queries['one_neg']
    for formula_type in data_queries:
        formulas = data_queries[formula_type]
        for formula in formulas:
            for q in formulas[formula]:
                ex = dict()
                ex['target'] = q.target_node
                ex['anchors'] = q.anchor_nodes
                ex['path'] = gen_path(q)
                ex['neg_samples'] = [] if is_train else q.neg_samples
                ex['type'] = formula_type
                if q.hard_neg_samples is not None or is_train:
                    ex['hard_negs'] = q.hard_neg_samples
                else:
                    ex['hard_negs'] = []
                data.append(ex)
    with open(data_dir,'w',encoding='utf8') as f:
        json.dump(data,f,indent=None,separators=(",\n",": "))

if __name__ == "__main__":
    process_bio()