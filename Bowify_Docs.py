import os
import re
import string
import pickle
import xml.etree.ElementTree as ET
from nltk.stem.snowball import SnowballStemmer

path = 'data/dataset101-150/'
data_dir = os.listdir(path)
stemmer = SnowballStemmer("english")
stop_words_file = open('data/common-english-words.txt', 'r').read()
stop_words_list = stop_words_file.split(",")



class BowDocument:
    def __init__(self, docID, datasetID):
        self.documentID = docID
        self.term_map = {}
        self.normalised_term_frequencies = {}
        self.docLength = 0
        self.dataset = datasetID

    def get_doc_id(self):
        print(self.documentID)

    def add_term(self, term):
        if not stop_words_list.__contains__(term):
            term = stem_word_by_snowball(term)
            if not (term == ''):
                    if self.term_map.__contains__(term):
                        self.term_map[term] += 1
                    else:
                        self.term_map[term] = 1

def stem_word_by_snowball(word):
    if len(word) >= 3:
        return stemmer.stem(word)
    return ''

def parse_raw_docs(subset_path, files):
    global terms, term
    documents = {}
    for filename in files:
        filepath = os.path.join(subset_path, filename)
        if not os.path.isdir(filepath) and not filename.startswith('._'):
            root = ET.parse(filepath).getroot()
            itemid = root.get('itemid')
            documents[itemid] = BowDocument(itemid, subset_path[-3:])
            for child in root.iter('p'):
                terms = [word.strip(string.punctuation) for word in child.text.split(" ")]
                for term in terms:
                    term = re.sub(r'[^a-zA-Z]', '', term)
                    documents[itemid].docLength += 1
                    documents[itemid].add_term(term)
    return documents

def normalise_term_frequencies(documents):
    for key, doc in documents.items():
        doc.normalised_term_frequencies = doc.term_map.copy()
        for key, value in doc.normalised_term_frequencies.items():
            doc.normalised_term_frequencies[key] = float(value)/float(doc.docLength)
    return documents

def pickle_bow_docs(subset_path, documents):
    for bow_doc in documents.values():
        directory = subset_path + '/pickles/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + bow_doc.documentID + '.pkl', 'wb') as output:
            pickle.dump(bow_doc, output, -1)


def walk_through_dataset():
    for root, dirs, files in os.walk(path):
        if not root.endswith('pickles') and not root.endswith('101-150/'):
            bowdocs = parse_raw_docs(root, files)
            bowdocs = normalise_term_frequencies(bowdocs)
            pickle_bow_docs(root, bowdocs)

def load_bowdocs():
    documents = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('pkl'):
                with open(root + '/' + file, 'rb') as input:
                    doc = pickle.load(input)
                    documents[doc.documentID] = doc
                    input.close()
    return documents

if __name__ == "__main__":
    walk_through_dataset()
    pass
