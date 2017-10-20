import os
import xml.etree.ElementTree as ET

path = 'data/dataset101-150/Training101/'
doc_dir = os.listdir(path)
topic_dir = os.listdir('data')


topic_statement_file = open('data/TopicStatements101-150.txt', 'r').readlines()
topic_statements = topic_statement_file
pass
