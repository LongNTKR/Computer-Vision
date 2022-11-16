
import re
import math
import random
import numpy as np
import pandas as pd
import nltk


def split_to_sentences(data):
    sentences = nltk.sent_tokenize(data)
    return sentences

def standardize_token(token):
    res = ''
    for i in range(len(token)):
        res += token[i] + ' '
    return res.strip()

def standardize_sentence(sentences):
    standard_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokenized = nltk.word_tokenize(sentence)
        standard_sentences.append(standardize_token(tokenized))
    return standard_sentences

def get_data_sentences():
    with open(r'clean_data.txt', 'r+', encoding='utf-8') as file:
        data = file.readlines()
        file.close()
    return data

def clean_data():
    with open(r'data.txt', 'r+', encoding='utf-8') as file:
        data = file.read()
        file.close()
    data = re.sub(r'\[\d+\]', '', data)
    sentences = split_to_sentences(data)
    standardized_sentence = standardize_sentence(sentences)
    clean_data = get_data_sentences() 
    with open(r'clean_data.txt', 'a', encoding='utf-8') as file:
        for i in standardized_sentence:
            if i not in clean_data:
                file.write(i + '\n')
        file.close()
clean_data()










