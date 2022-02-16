def get_specter_abstracts_dict(path = '/content/CitationPredictionChallenge/Assets/abstracts.txt'):
    abstracts = dict()
    with open(path, 'r', encoding="latin-1") as f:
        for line in f:
            node, full_abstract = line.split('|--|')
            abstract_list = full_abstract.split('.')
            full_abstract = '[SEP]'.join(abstract_list)

            abstracts[int(node)] = full_abstract
    return abstracts