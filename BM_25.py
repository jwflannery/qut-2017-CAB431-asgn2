import os
import pickle
import string
from operator import itemgetter
import re
from Bowify_Docs import BowDocument
from collections import OrderedDict

path = 'data/dataset101-150/'

def load_training_data():
    train_sets = {}
    training_path = 'data/trainingset101-150'
    training_directory = os.listdir(training_path)
    for setfile in training_directory:
        set = re.findall(r'\d+', setfile)[0]
        if not set in train_sets:
            train_sets[set] = {}
        set_lines = open(training_path + '/' + setfile, 'r').read().splitlines()
        for line in set_lines:
            train_sets[set][line.split()[1]] = line.split()[2]
        pass
    return train_sets

def load_idf_map():
    with open(path + 'idf_map.pkl', 'rb') as input:
        idf_map = pickle.load(input)
        input.close()
    return idf_map

def load_bowdocs():
    documents = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('pkl') and not file.startswith('idf_map'):
                with open(root + '/' + file, 'rb') as input:
                    doc = pickle.load(input)
                    if not doc.dataset in documents:
                        documents[doc.dataset] = {}
                    documents[doc.dataset][doc.documentID] = doc
                    input.close()
    return documents

def get_relevant(train_sets):
    relevant_docs = {}
    for key, set in train_sets.items():
        if not key in relevant_docs:
            relevant_docs[key] = {}
        relevant_docs[key] = {k: v for k, v in set.items() if v == '1'}
    return relevant_docs

def get_non_relevant(train_sets):
    non_relevant_docs = {}
    for key, set in train_sets.items():
        if not key in non_relevant_docs:
            non_relevant_docs[key] = {}
        non_relevant_docs[key] = {k: v for k, v in set.items() if v == '0'}
    return non_relevant_docs

def BM25_Step1(bowdocs):
    terms = []
    for key, doc in bowdocs.items():
        for term, freq in doc.term_map.items():
            if not term in terms:
                terms.append(term)
    return terms

def BM25_Step2(terms):
    n_tk = {}
    r_tk = {}
    for term in terms:
        n_tk[term] = 0
        r_tk[term] = 0
    return n_tk, r_tk


def BM25_Step3(n_tk, docs):
    for term, n in n_tk.items():
        for key, doc in docs.items():
            if term in doc.term_map:
                n_tk[term] += 1
    return n_tk

def BM25_Step4(r_tk, rel_docs):
    for term, n in r_tk.items():
        for key, doc in rel_docs.items():
            if term in doc.term_map:
                r_tk[term] += 1
    return r_tk

def BM25_Step5(terms, r_tk, n_tk, N, R):
    weights = {}
    for term in terms:
        rtk = r_tk[term]
        ntk = n_tk[term]
        weight = (rtk + 0.5) / (R - rtk + 0.5)
        weight = weight / ((ntk - rtk + 0.5) / ((N - ntk) - (R - rtk) + 0.5))
        weights[term] = weight
    return weights

def BM25(bowdocs, rel_docs, non_rel_docs):
    weights = {}
    for key, set in bowdocs.items():
        terms = BM25_Step1(bowdocs[key])
        n_tk, r_tk = BM25_Step2(terms)
        n_tk = BM25_Step3(n_tk, bowdocs[key])
        r_tk = BM25_Step4(r_tk, rel_docs[key])
        weights[key] = BM25_Step5(terms, r_tk, n_tk, len(bowdocs[key]), len(rel_docs[key]))
    return weights

def ID_to_doc(IDs, bowdocs):
    docs = {}
    for key, set in bowdocs.items():
        for docID, bow_doc in set.items():
            if not bow_doc.dataset in docs:
                docs[bow_doc.dataset] = {}
            if bow_doc.documentID in IDs[bow_doc.dataset]:
                docs[bow_doc.dataset][bow_doc.documentID] = bow_doc
    return docs

def rank_documents(weights):
    doc_scores = {}
    for setID, set in bowdocs.items():
        if not setID in doc_scores:
            doc_scores[setID] = {}
        for key, doc in bowdocs[setID].items():
            doc_scores[setID][key] = 0
            for term, freq in doc.term_map.items():
                doc_scores[setID][key] += weights[setID][term]
    return doc_scores

def output_ranks(rankings):
    for setID, set in rankings.items():
        file = open("data/IFresults101-150/result" + setID[1:] + ".dat", 'w')
        for docID, score in sorted(rankings[setID].items(), key=itemgetter(1), reverse=True):
            file.write(docID + ' ')
            file.write(str(score) + '\n')
    pass


if __name__ == "__main__":
    bowdocs = load_bowdocs()
    idf_map = load_idf_map()
    train_sets = load_training_data()

    relevant_IDs = get_relevant(train_sets)
    non_relevant_IDs = get_non_relevant(train_sets)
    relevant_docs = ID_to_doc(relevant_IDs, bowdocs)
    non_relevant_docs = ID_to_doc(non_relevant_IDs, bowdocs)

    weights = BM25(bowdocs, relevant_docs, non_relevant_docs)

    ranks = rank_documents(weights)
    output_ranks(ranks)
    pass