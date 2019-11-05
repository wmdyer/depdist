import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys

def load_ud_file(filename):
    ud = pd.read_csv(filename, sep="\t",error_bad_lines=False, engine='python', header=None, comment= '#', quoting=3)
    ud.columns = ['idx', 'lemma', 'wordform', 'ud_pos', 'sd_pos', 'morph', 'head', 'rel1', 'rel2', 'other']
    return ud

def load_vectors(filename):
    df = pd.read_csv(filename, sep='["]* ["]*', header=None, error_bad_lines=False, engine='python')
    return df

def get_word_vector(word, vecs):
    try:
        return vecs.loc[vecs[0] == word].values[0][1:]
    except:
        return np.zeros(vecs.shape[1]-1)

def one_hot(codes, values):
    code = np.zeros(len(codes))
    for val in values:
        code[codes.index(val)] = 1
    return code

def unpack(pack_values, sep):
    values = []
    for pack in pack_values:
        for val in str(pack).split(sep):
            values.append(val)
    return list(set(values))

def make_node_vecs(ud, wv):
    ud_pos_vals = list(ud.ud_pos.unique())
    morph_vals = unpack(list(ud.morph.unique()), '|')
    rel_vals = unpack(list(ud.rel1.unique()), ':')    

    node_vecs = []
    for i, row in ud.iterrows():
        try:
            lemma = get_word_vector(row['lemma'].lower(), wv)
            wordform = get_word_vector(row['wordform'].lower(), wv)        
            ud_pos = one_hot(ud_pos_vals, [row['ud_pos']])
            morph = one_hot(morph_vals, row['morph'].split("|"))
            rel = one_hot(rel_vals, row['rel1'].split(":"))            
            #print(row['morph'].split("|"))
            node_vecs.append(np.concatenate([lemma, wordform, ud_pos, morph, rel]))
        except:
            if row['lemma'] is None or row['wordform'] is None:
                pass
            else:
                print(row)
                exit()
        if i > 20:
            break
    return np.array(node_vecs)

def make_graphs_nx(ud, node_vecs):
    g = nx.DiGraph()
    for i,row in ud.iterrows():
        g.add_node(str(int(row['idx']) + i), features=node_vecs[i])
        g.add_edge(str(int(row['head']) + i), str(int(row['idx']) + i), features=np.array([0.0]))
        if i > 20:
            break
    return g

if __name__ == '__main__':
    print("loading " + sys.argv[1])
    ud = load_ud_file(sys.argv[1])

    print("loading " + sys.argv[2])
    vectors = load_vectors(sys.argv[2])

    print("making node_vecs")
    node_vecs = make_node_vecs(ud, vectors)

    print("making graphs_nx")
    graphs_nx = make_graphs_nx(ud, node_vecs)
    ax = plt.figure(figsize=(3,3)).gca()
    nx.draw(graphs_nx, ax=ax)
    plt.savefig("graph.png")
    print(graphs_nx)
    #graphs_tuple = utils_np.networkxs_to_graphs_tuple([graphs_nx])

    
