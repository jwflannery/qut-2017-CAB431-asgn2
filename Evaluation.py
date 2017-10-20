import os
import re
from operator import itemgetter
from collections import Counter
from collections import OrderedDict
import csv

def load_topic_assignments():
    topic_assignments = {}
    topic_path = 'data/topicassignment101-150'
    topic_directory = os.listdir(topic_path)
    for topicfile in topic_directory:
        topic_lines = open(topic_path + '/' + topicfile, 'r').read().splitlines()
        set = re.findall(r'\d+', topicfile)[0]
        if not set in topic_assignments:
            topic_assignments[set] = {}
        for line in topic_lines:
            topic_assignments[set][line.split()[1]] = line.split()[2]
    return topic_assignments

def load_IR_results():
    IR_results = {}
    IR_path = 'data/IRresults101-150'
    IR_directory = os.listdir(IR_path)
    for resultFile in IR_directory:
        result_lines = open(IR_path + '/' + resultFile, 'r').read().splitlines()
        set = re.findall(r'\d+', resultFile)[0]
        if not set in IR_results:
            IR_results[set] = {}
        for line in result_lines:
            IR_results[set][line.split()[0]] = float(line.split()[1])
    for set, results in IR_results.items():
        IR_results[set] = OrderedDict(IR_results[set].items(), key=itemgetter(1), reverse=True)
    return IR_results

def load_IF_results():
    IF_results = {}
    IF_path = 'data/IFresults101-150'
    IF_directory = os.listdir(IF_path)
    for resultFile in IF_directory:
        result_lines = open(IF_path + '/' + resultFile, 'r').read().splitlines()
        set = re.findall(r'\d+', resultFile)[0]
        if not set in IF_results:
            IF_results[set] = {}
        for line in result_lines:
            IF_results[set][line.split()[0]] = float(line.split()[1])
    for set, results in IF_results.items():
        IF_results[set] = OrderedDict(IF_results[set].items(), key=itemgetter(1), reverse=True)
    return IF_results

def top10_results(result_set, test_set):
    top10 = {}
    for key, dataset in result_set.items():
        if not key in top10:
            top10['1' + key] = {}
        top10['1' + key] = (list(dataset.items())[:10])
    return top10

def calc_precision(results, test_set, n):
    precision = {}
    for key, dataset in results.items():
        i = 0
        j = 0
        if not key in precision:
            precision[key] = 0
        for doc, score in dataset:
            i += 1
            if test_set[key][doc] == '1':
                j += 1
                precision[key] += j / i
        precision[key] = precision[key] / n
    return precision

def calc_recall(results, test_set):
    recall = {}
    n = 0
    for key, dataset in results.items():
        if not key in recall:
            recall[key] = 0
        for doc, score in dataset:
            if test_set[key][doc] == '1':
                recall[key] += 1
        for doc, rel in test_set[key].items():
            if rel == '1':
                n += 1
        recall[key] = recall[key] / n
    return recall

def calc_Fmeasure(precision, recall, setlist):
    F = {}
    for key in setlist:
        if not recall[key] + precision[key] == 0:
            F[key] = (2*recall[key]*precision[key]) / (recall[key] + precision[key])
        else:
            F[key] = 0
    return F

def output_eval(measure1, measure2, filename):
    with open('data/' + filename + '.csv', 'w', newline='') as output:
        #writer = csv.writer(output, delimiter=' ', quotechar = '|', quoting=csv.QUOTE_MINIMAL)
        fieldnames = ['IR Score', 'IF Score']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in measure1.items():
            writer.writerow({'IR Score': value, 'IF Score': measure2[key]})


if __name__ == "__main__":
    test_set = load_topic_assignments()
    IR_results = load_IR_results()
    IF_results = load_IF_results()

    IR_top10 = top10_results(IR_results, test_set)
    IF_top10 = top10_results(IF_results, test_set)

    IRprecision = calc_precision(IR_top10, test_set, 10.0)
    IFprecision = calc_precision(IF_top10, test_set, 10.0)
    IRrecall = calc_recall(IR_top10, test_set)
    IFrecall = calc_recall(IF_top10, test_set)

    total_IR_prec = sum(Counter(IRprecision).values())
    total_IF_prec = sum(Counter(IFprecision).values())
    total_IR_recall = sum(Counter(IRrecall).values())
    total_IF_recall = sum(Counter(IFrecall).values())

    IR_Fmeasure = calc_Fmeasure(IRprecision, IRrecall, test_set.keys())
    IF_Fmeasure = calc_Fmeasure(IFprecision, IFrecall, test_set.keys())

    total_IR_Fmeasure = sum(Counter(IR_Fmeasure).values())
    total_IF_Fmeasure = sum(Counter(IF_Fmeasure).values())

    IR_MAP = (sum(Counter(IRprecision).values()))/len(IRprecision)
    IF_MAP = (sum(Counter(IFprecision).values()))/len(IFprecision)

    output_eval(IR_Fmeasure, IF_Fmeasure, 'evaluation_F-measure')
    output_eval(IRprecision, IFprecision, 'evaluation_top10')



    pass