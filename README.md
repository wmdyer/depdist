# DepDist
Learn dependency distance tolerances [(Dyer, 2019)](https://www.aclweb.org/anthology/W19-7807.pdf) from Universal Dependencies conllu files and generate surface orders. The learning process uses a Graph Neural Network that's fed a series of networkx graphs in which each node's attribute is a vectorized representation of a word consisting of an GloVe embedding and one-hot encodings of other conllu data (e.g., part of speech and relation type). Message-passing allows nodes to know about each other, allowing the system to learn context-sensitive dependency distances. The edge weights, representing dependency distance tolerances, are learned by the GNN and reported as edge attributes.

![GNN](/img/input.png)

After edge weights are learned, a surface realization is generated without any sort of projectivity constraint.

![surface realization](/img/output.png)

## Data

### External
* conllu train/dev/test files from [Universal Dependencies](https://github.com/UniversalDependencies/)
* word vectors (e.g., http://nlp.stanford.edu/data/glove.42B.300d.zip)

## Learn
`python learn.py -u <one or more directories containing train/dev/test conllu files> -v <file containing word vectors>`

Output is a series of `distance_<epoch>.<conllu>.dev` files containing dependency distance tolerances between dependents and heads, named according to training epoch and conllu directory. Each tab-delimited output file contains the following columns:
1. `wordform`
1. `sentence_id`
1. `word_id`
1. `head_id`
1. `distance`

## Sort (non-projective)
`python npsort.py <distance_file> <output_file>`

When given a file containing distances, `npsort.py` will generate an output file in the following format:
```
# sent_id = 1
# text = a hearing is scheduled on the issue tomorrow
```

