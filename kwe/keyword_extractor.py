#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Keyword extraction module for KWE."""

from kwe.tokenizer import KeywordTokenizer

class KeywordExtractor(object):

    def __init__(self, input_file, max_keyword_size=3):
        self.input_file = input_file
        self.max_keyword_size = max_keyword_size

    def extract_keywords(self):
        sentences = KeywordTokenizer.tokenize_sentences(self.input_file)
        candidate_keywords = list(KeywordTokenizer.tokenize_keywords(
            sentences, self.max_keyword_size
        ))
