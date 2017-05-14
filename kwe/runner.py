#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Runner module for KWE."""

from kwe.tokenizer import KeywordTokenizer

def extract_keywords(input_file):
    sentences = KeywordTokenizer.tokenize_sentences(input_file)
    candidate_keywords = KeywordTokenizer.tokenize_keywords(sentences)
    kws = list(candidate_keywords)

if __name__ == '__main__':
    extract_keywords('data/script.txt')
