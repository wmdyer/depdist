import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
import pandas as pd
import networkx as nx
import numpy as np
import sys, operator
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(filename):
    df = pd.read_csv(str(filename), sep="\t",error_bad_lines=False, engine='python', comment= '# ', quoting=3)
    return df

def print_progress(i, n):
    j = (i) / n
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%% %d/%d" % ('='*int(20*j), 100*j, i, n))
    sys.stdout.flush()
    return i +1

def linearize(df, sid):
    df = update_indexes(df, 0)
    sentence = ""
    for word in df.sort_values(by=['idx'])['word'].values:
        sentence+=word + " "
    return sentence

def update_indexes(df, head):
    deps = df.loc[df['hid'] == head]
    for d,dep in deps.iterrows():
        try:
            hidx = df.loc[df['wid'] == head]['idx'].values[0]
        except:
            hidx = 0
        if head == 0:
            dist = 0
        else:
            dist = dep['pdist']
        df.loc[df['wid'] == dep['wid'], 'idx'] += hidx + dist
        if len(df.loc[df['hid'] == head]) > 0:
            update_indexes(df, dep['wid'])
    return df

input_file = sys.argv[1]
output_file = sys.argv[2]

df = load_data(input_file)
df['idx'] = np.zeros(df.shape[0])
total = len(df.sid.unique())
predict_output = []

for sid in df.sid.unique():
    print_progress(int(sid), total)
    dfa = df[df['sid'] == sid][['word', 'wid', 'hid', 'pdist', 'idx']]
    if dfa.size > 3:
        predict = linearize(dfa, sid)
        predict_output.append(predict)
print("")

outfile = open(output_file, "w")
for i,p in enumerate(predict_output):
    outfile.write("#sent_id = " + str(i+1) + "\n")
    outfile.write("#text = " + p + "\n")
    outfile.write("\n")
outfile.close()

