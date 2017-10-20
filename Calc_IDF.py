import os
import pickle
import math
from Bowify_Docs import BowDocument
from collections import OrderedDict

path = 'data/dataset101-150/'

def load_bowdocs():
    documents = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('pkl') and not file.startswith('idf_map'):
                with open(root + '/' + file, 'rb') as input:
                    doc = pickle.load(input)
                    documents[doc.documentID] = doc
                    input.close()
    return documents

def calculate_df(documents):
    df_map = {}
    for key, doc in documents.items():
        for term, freq in doc.term_map.items():
            if doc.dataset in df_map:
                if term in df_map[doc.dataset]:
                    df_map[doc.dataset][term] += 1
                else:
                    df_map[doc.dataset][term] = 1
            else:
                df_map[doc.dataset] = {}
    return df_map

def calculate_idf(df_map):
    idf_map = df_map.copy()
    for key, dataset in df_map.items():
        for term, freq in dataset.items():
            idf_map[key][term] = math.log(len(dataset)/freq)
    return idf_map

def pickle_idf_map(map):
    directory = path
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path + 'idf_map.pkl', 'wb') as output:
        pickle.dump(map, output, -1)

if __name__ == "__main__":
    documents = load_bowdocs()
    df_map = calculate_df(documents)
    idf_map = calculate_idf(df_map)
    pickle_idf_map(idf_map)
    pass