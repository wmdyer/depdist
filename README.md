# DepDist
Learn dependency distance tolerances [(Dyer, 2019)](https://www.aclweb.org/anthology/W19-7807.pdf) from Universal Dependencies conllu files and generate surface orders. The learning process uses a Graph Neural Network that's fed a series of networkx graphs in which each node's attribute is a vectorized representation of a word consisting of an GloVe embedding and one-hot encodings of other conllu data (e.g., part of speech and relation type). Message-passing allows nodes to know about each other, allowing the system to learn context-sensitive dependency distances. The edge weights, representing dependency distance tolerances, are learned by the GNN and reported as edge attributes.

![GNN](/img/input.png)

After edge weights are learned, a surface realization is generated without any sort of projectivity constraint.

![surface realization](/img/output.png)

## Background

Dyer, W. (2019). DepDist: Surface realization via regex and dependency distance tolerance. In *Proceedings of the Second Workshop on Multilingual Surface Realisation (SR'19)*. Hong Kong. Association for Computational Linguistics. [ACL link](https://www.aclweb.org/anthology/D19-6303.pdf)

Dyer, W. (2019). Weighted posets: Learning surface order from dependency structure. In *Proceedings of the 18th International Workshop on Treebanks and Linguistic Theories (TLT, SyntaxFest 2019)*, pp. 61-73. Paris, France. Association for Computational Linguistics. [ACL link](https://www.aclweb.org/anthology/W19-7807.pdf)

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

## Results
At the Second Multilingual Surface Realization Shared Task in Hong Kong, Nov 3, 2019 [(Mille et al., 2019)](https://www.aclweb.org/anthology/D19-63.pdf), the DepDist system performed in the middle of the other submissions.

![shared task](/img/results.png)

NB: The shared task included inflection, a subtask not included in this repo.
