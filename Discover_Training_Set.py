import os
import string
import pickle
import re
from operator import itemgetter
from Bowify_Docs import BowDocument
from collections import OrderedDict
from collections import Counter
from nltk.stem.snowball import SnowballStemmer

path = 'data/dataset101-150/'
stemmer = SnowballStemmer("english")
stop_words_file = open('data/common-english-words.txt', 'r').read()
stop_words_list = stop_words_file.split(",")
topic_lines = open('data/TopicStatements101-150.txt').read().splitlines()
scores = {}

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

def stem_word_by_snowball(word):
    if len(word) >= 3:
        return stemmer.stem(word)
    return ''

def parse_raw_query(query):
    processed_query = []
    query_terms = [word.strip(string.punctuation) for word in query.split(" ")]
    for term in query_terms:
        term = re.sub(r'[^a-zA-Z]', '', term)
        if not term in stop_words_list:
            term = stem_word_by_snowball(term)
            if not term == '':
                processed_query.append(term)
    return processed_query

def calc_tf_idf(term, document):
    tf_idf = 0
    if term in document.normalised_term_frequencies and term in idf_map[document.dataset]:
        tf_idf = document.normalised_term_frequencies[term] * idf_map[document.dataset][term]
    return tf_idf

def calc_doc_tf_idf_score(termlist, document):
    score = 0
    for term in termlist:
        score += calc_tf_idf(term, document)
    return score

def calc_dataset_scores(doclist, raw_query):
    clean_query = parse_raw_query(raw_query)
    for key, doc in doclist.items():
        if not doc.dataset in scores:
            scores[doc.dataset] = {}
        scores[doc.dataset][doc.documentID] = calc_doc_tf_idf_score(clean_query, doc)
    return scores

def parse_titles(topic_lines):
    titles = {}
    topic_num = '0'
    for line in topic_lines:
        if line.startswith('<num>'):
            topic_num = line.strip()[-3:]
        if line.startswith('<title>'):
            titles[topic_num] = line[7:]
        #if not line.startswith('<') and not line.strip() == '':
        #    titles[topic_num] += ' ' + (line.strip())
    return titles

def calc_all_scores(bow_docs, queries):
    for key, query in queries.items():
        calc_dataset_scores(bow_docs[key], query)

def get_top_scores(scores):
    top_scores = {}
    for key, dataset in scores.items():
        top_scores[key] = dict(Counter(scores[key]).most_common(5))
    return top_scores

def create_training_set(top_scores, all_scores):
    for set, docs in all_scores.items():
        file = open("data/trainingset101-150/Train" + set + ".txt", 'w')
        for key, doc in docs.items():
            file.write(set + ' ')
            file.write(key + ' ')
            if key in top_scores[set]:
                file.write('1\n')
            else:
                file.write('0\n')

def output_ranks(all_scores):
    for setID, set in all_scores.items():
        file = open("data/IRresults101-150/BaselineResult" + setID[1:] + ".dat", 'w')
        for docID, score in sorted(all_scores[setID].items(), key=itemgetter(1), reverse=True):
            file.write(docID + ' ')
            file.write(str(score) + '\n')
    pass

if __name__ == "__main__":
    bow_docs = load_bowdocs()
    idf_map = load_idf_map()

    raw_queries = parse_titles(topic_lines)
    calc_all_scores(bow_docs, raw_queries)

    top_scores = get_top_scores(scores)
    create_training_set(top_scores, scores)

    output_ranks(scores)
    pass
