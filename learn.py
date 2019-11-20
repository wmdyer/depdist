import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sonnet as snt
import tensorflow as tf
import sys, pickle, math, argparse

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

#from sklearn.cluster import KMeans

import models

tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

verbose = False

def print_progress(e, i, n, s):
    j = (i) / n
    sys.stdout.write('\r')
    sys.stdout.write("Epoch " + str(e) + " [%-20s] %d%% batch %d/%d Ltr %f" % ('='*int(20*j), 100*j, i, n, s))
    sys.stdout.flush()
    return i + 1

def load_ud_file(filename):
    ud = pd.read_csv(filename, sep="\t",error_bad_lines=False, engine='python', header=None, comment= '# ', quoting=3)
    ud.columns = ['idx', 'wordform', 'lemma', 'ud_pos', 'sd_pos', 'morph', 'head', 'rel1', 'rel2', 'other']
    return ud

def load_vectors(filename):
    df = pd.read_csv(filename, sep='["]* ["]*', header=None, error_bad_lines=False, engine='python', index_col=0)
    x = run_pca(df.values)
    w = df.index
    df = pd.concat([pd.DataFrame(w), pd.DataFrame(x)], axis=1, ignore_index=True)
    df.set_index([0], inplace=True)
    glove = {key: val.values for key, val in df.T.items()}
    return glove

def get_word_vector(word, vecs):
    try:
        return vecs.loc[vecs[0] == word].values[0][1:]
    except:
        return np.zeros(vecs.shape[1]-1)
    
def run_pca(x):
    xdim = x.shape[1]
    pca = PCA(0.5)
    x = StandardScaler().fit_transform(x)
    principalComponents = pca.fit_transform(x)
    print("pca: " + str(xdim) + " -> " + str(pca.n_components_))
    return principalComponents

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

def make_node_vecs(row, glove, uniq_vals):
    vec_length = len(glove['the'])

    try:
        lemma = glove[row['lemma'].lower()]
    except:
        lemma = np.zeros(vec_length)
    #try:
    #    wordform = glove[row['wordform'].lower()]
    #except:
    #    wordform = np.zeros(vec_length)
    ud_pos = one_hot(uniq_vals['ud_pos'], [row['ud_pos']])
    sd_pos = one_hot(uniq_vals['sd_pos'], [row['sd_pos']])
    morph = one_hot(uniq_vals['morph'], row['morph'].split("|"))
    rel = one_hot(uniq_vals['rel'], row['rel1'].split(":"))

    vector = np.array(np.concatenate([lemma, ud_pos, sd_pos, morph, rel]))
    return vector, vector
    
def make_graphs_nx(ud, glove, uniq_vals):
    input_graphs = []
    target_graphs = []
    input_g = None
    target_g = None
    n = ud.shape[0]
    sid = 0
    for i,row in ud.iterrows():
        #print_progress(i+1, n)
        try:
            idx = int(row['idx'])
            head = int(row['head'])
            input_node_vec, target_node_vec = make_node_vecs(row, glove, uniq_vals)
            if idx == 1:
                sid += 1
                if input_g is not None:
                    input_graphs.append(input_g)
                    target_graphs.append(target_g)
                input_g = nx.DiGraph()
                input_g.graph["features"] = np.array([0.])
                input_g.add_node(0, features=np.zeros(len(input_node_vec)), dtype=np.float)

                target_g = nx.DiGraph()
                target_g.graph["features"] = np.array([0.])
                target_g.add_node(0, features=np.zeros(len(target_node_vec)), dtype=np.float)
            input_g.add_node(idx, features=input_node_vec, dtype=np.float)
            target_g.add_node(idx, features=target_node_vec)
            
            input_g.add_edge(idx, head, features=np.array([0.0]))
            target_g.add_edge(idx, head, features=np.array([idx-head]))
            
            input_g.add_edge(head, idx, features=np.array([0.0]))
            target_g.add_edge(head, idx, features=np.array([idx-head]))            
        except Exception as e:
            pass
            #print(e)
            #print(input_g.nodes)
            #print(input_g.edges)
            #print(row)
            #print(sid)
            #exit()

    input_graphs.append(input_g)
    target_graphs.append(target_g)    
    return {'input': input_graphs, 'target': target_graphs}

def build_graphs_tuples(ud_locs, glove_loc):
    ud_train = pd.DataFrame()
    ud_test = pd.DataFrame()
    ud_dev = pd.DataFrame()
    
    for ud_loc in ud_locs:
        ud_files = os.listdir(ud_loc)
    
        train_file = ud_loc + "/" + str(list(filter(lambda x:'train.conllu' in x, ud_files))[0])
        print("loading " + train_file)
        train = load_ud_file(train_file)
        train['corpus'] = train_file
        ud_train = pd.concat([ud_train, train], axis=0, ignore_index=True, sort=False)
    
        dev_file = ud_loc + "/" + str(list(filter(lambda x:'dev.conllu' in x, ud_files))[0])
        print("loading " + dev_file)
        dev = load_ud_file(dev_file)
        dev['corpus'] = dev_file
        ud_dev = pd.concat([ud_dev, dev], axis=0, ignore_index=True, sort=False)
    
        test_file = ud_loc + "/" + str(list(filter(lambda x:'test.conllu' in x, ud_files))[0])
        print("loading " + test_file)
        test = load_ud_file(test_file)
        test['corpus'] = test_file
        ud_test = pd.concat([ud_test, test], ignore_index=True, sort=False)
    
    print("extracting uniq values")
    ud_all = pd.concat([ud_train, ud_dev, ud_test])
    uniq_vals = {}
    uniq_vals['ud_pos'] = list(ud_all.ud_pos.unique())
    uniq_vals['sd_pos'] = list(ud_all.sd_pos.unique())
    uniq_vals['morph'] = unpack(list(ud_all.morph.unique()), '|')
    uniq_vals['rel'] = unpack(list(ud_all.rel1.unique()), ':')    
    
    print("loading word vectors from " + str(glove_loc))
    glove = load_vectors(glove_loc)
        
    print("making graphs_nx")
    train_graphs = {}
    dev_graphs = {}
    test_graphs = {}
    train_graphs = make_graphs_nx(ud_train, glove, uniq_vals)
    dev_graphs = make_graphs_nx(ud_dev, glove, uniq_vals)
    test_graphs = make_graphs_nx(ud_test, glove, uniq_vals)
    
    print("\nconverting to graphs_tuples")
    train_tuples = {}
    dev_tuples = {}
    test_tuples = {}
    for gtype in ['input', 'target']:
        train_tuples[gtype] = utils_np.networkxs_to_graphs_tuple(train_graphs[gtype])
        dev_tuples[gtype] = utils_np.networkxs_to_graphs_tuple(dev_graphs[gtype])
        test_tuples[gtype] = utils_np.networkxs_to_graphs_tuple(test_graphs[gtype])
    
    print("saving to " + pkl_file)
    f = open(pkl_file, 'wb')
    pickle.dump(ud_all, f)
    pickle.dump(train_graphs, f)
    pickle.dump(dev_graphs, f)
    pickle.dump(test_graphs, f)
    pickle.dump(train_tuples, f)
    pickle.dump(dev_tuples, f)
    pickle.dump(test_tuples, f)
    f.close()

def create_placeholders(input_graphs, target_graphs):
    input_ph = utils_tf.placeholders_from_networkxs(input_graphs)
    target_ph = utils_tf.placeholders_from_networkxs(target_graphs)
    return input_ph, target_ph

def create_loss_ops(target_op, output_ops):
    loss_ops = [
        tf.compat.v1.losses.absolute_difference(target_op.edges, output_op.edges)
        #tf.compat.v1.losses.absolute_difference(target_op.nodes, output_op.nodes)        
    for output_op in output_ops
    ]
    return loss_ops

def output_dist_file(ud, values, epoch, otype):
    outfile = open("distance_" + str(epoch) + "." + str(otype), "w")
    outfile.write("word\tsid\twid\thid\tadist\tpdist\n")

    sid = 0
    for i,row in ud.iterrows():
        if row['idx'] == 1:
            sid+=1
        pdist = values.edges[i][0]
        wordform = row['wordform']
        try:
            outfile.write(wordform + "\t" + str(sid) + "\t" + str(row['idx']) + "\t" + str(row['head']) + "\t" + str(int(row['idx']) - int(row['head'])) + "\t" + str(pdist) + "\n")
        except:
            pass
    outfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='learn dep distances')
    parser.add_argument('-u','--ud', nargs='+', dest='ud', help='<Required> directories containing conllu files', required=True)
    parser.add_argument('-v','--vectors', nargs=1, dest='vectors', help='<Required> file containing word vectors', required=True)
    args = parser.parse_args()
    
    pkl_file = "data.pkl"
    if pkl_file not in os.listdir():
        build_graphs_tuples(args.ud, args.vectors[0])
        
    print("loading from " + pkl_file)
    f = open(pkl_file, 'rb')
    ud_all = pickle.load(f)
    train_graphs = pickle.load(f)
    dev_graphs = pickle.load(f)
    test_graphs = pickle.load(f)
    train_tuples = pickle.load(f)
    dev_tuples = pickle.load(f)
    test_tuples = pickle.load(f)    
    f.close()

    n_train = train_tuples['input'].n_node.shape[0]
    print("train: " + str(n_train))
    print("  dev: " + str(dev_tuples['input'].n_node.shape[0]))
    print(" test: " + str(test_tuples['input'].n_node.shape[0]))

    BATCH_SIZE = 128
    NUM_MPNN = 3
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    NUM_TRAINING_ITERS = 100

    num_sub = math.ceil(n_train/BATCH_SIZE)

    OUTPUT_EDGE_SIZE = train_tuples['target'].edges.shape[1]
    OUTPUT_NODE_SIZE = train_tuples['target'].nodes.shape[1]
    OUTPUT_GLOBAL_SIZE = 0    

    tf.reset_default_graph()

    model = models.EncodeProcessDecode(edge_output_size=OUTPUT_EDGE_SIZE, node_output_size=OUTPUT_NODE_SIZE, global_output_size=OUTPUT_GLOBAL_SIZE)

    optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE)

    input_ph = utils_tf.placeholders_from_networkxs(train_graphs['input'][0:1])
    target_ph = utils_tf.placeholders_from_networkxs(train_graphs['target'][0:1])
    output_ops = model(input_ph, NUM_MPNN)
    loss_ops = create_loss_ops(target_ph, output_ops)
    loss_op = sum(loss_ops)
    step_op = optimizer.minimize(loss_op)
        
    feed_dicts = []
    for i,batch in enumerate(range(0, n_train, BATCH_SIZE)):
        feed_dicts.append({
            input_ph: utils_np.networkxs_to_graphs_tuple(train_graphs['input'][batch:batch+BATCH_SIZE]),
            target_ph: utils_np.networkxs_to_graphs_tuple(train_graphs['target'][batch:batch+BATCH_SIZE])
        })

    input_ph_dev = utils_tf.placeholders_from_networkxs(dev_graphs['input'][0:1])
    target_ph_dev = utils_tf.placeholders_from_networkxs(dev_graphs['target'][0:1])
    output_ops_dev = model(input_ph_dev, NUM_MPNN)
    loss_ops_dev = create_loss_ops(target_ph_dev, output_ops_dev)
    loss_op_dev = loss_ops_dev[-1]
    feed_dict_dev = {input_ph_dev: dev_tuples['input'], target_ph_dev: dev_tuples['target']}

    input_ph_test = utils_tf.placeholders_from_networkxs(test_graphs['input'][0:1])
    target_ph_test = utils_tf.placeholders_from_networkxs(test_graphs['target'][0:1])
    output_ops_test = model(input_ph_test, NUM_MPNN)
    loss_ops_test = create_loss_ops(target_ph_test, output_ops_test)
    loss_op_test = loss_ops_test[-1]
    feed_dict_test = {input_ph_test: test_tuples['input'], target_ph_test: test_tuples['target']}    

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(0, NUM_EPOCHS):
        losses = []
        for i in range(0, len(feed_dicts)):
            for iteration in range(0, NUM_TRAINING_ITERS):
                train_values = sess.run({
                    "step": step_op,
                    "target": target_ph,
                    "loss": loss_op,
                    "outputs": output_ops
                    }, feed_dict = feed_dicts[i])
            losses.append(train_values["loss"])
            print_progress(epoch+1, i+1, len(feed_dicts), np.average(np.array(losses)))

        dev_values = sess.run({
            "target": target_ph_dev,
            "loss": loss_op_dev,
            "outputs": output_ops_dev
        }, feed_dict = feed_dict_dev)

        test_values = sess.run({
            "target": target_ph_test,
            "loss": loss_op_test,
            "outputs": output_ops_test
        }, feed_dict = feed_dict_test)            
            
        print(" Lde {:.4f} Lte {:.4f}".format(dev_values["loss"], test_values["loss"]))
        for ud in args.ud:
            output_dist_file(ud_all.loc[(ud_all['corpus'].str.contains("dev")) & (ud_all['corpus'].str.contains(ud))], dev_values["outputs"][-1], epoch, ud.replace("/", "").replace(".",""))
