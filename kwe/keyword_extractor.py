#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Keyword extraction module for KWE."""

import itertools
from collections import defaultdict

import pandas as pd

from kwe.tokenizers import RegexpKeywordTokenizer

class KeywordExtractor(object):
    """Keyword extraction class."""

    def __init__(self, input_file, max_keyword_size=3, tokenizer=None):
        self.input_file = input_file
        self.max_keyword_size = max_keyword_size
        self.co_occurrency_matrix = None
        self.all_words = None

        if tokenizer is None:
            self.tokenizer = RegexpKeywordTokenizer
        else:
            self.tokenizer = tokenizer

    def extract_keywords(self):
        candidate_keywords = self._extract_keyword_candidates()
        self._build_co_occurrency_matrix(candidate_keywords)
        self._compute_all_keyword_scores(candidate_keywords)

    def _extract_keyword_candidates(self):
        sentences = self.tokenizer.tokenize_sentences(self.input_file)
        candidate_keywords = list(self.tokenizer.tokenize_keywords(
            sentences, self.max_keyword_size
        ))

        return candidate_keywords

    def _build_co_occurrency_matrix(self, candidate_keywords):
        self.all_words = {word.lower() for word in itertools.chain.from_iterable(candidate_keywords)}
        self.co_occurrency_matrix = pd.DataFrame(
            0, index=self.all_words, columns=self.all_words
        )

        for kw in candidate_keywords:
            for w1 in kw:
                for w2 in kw:
                    self.co_occurrency_matrix[w1.lower()][w2.lower()] += 1

    def _compute_all_keyword_scores(self, candidate_keywords):
        # Create a pandas Series containing word degree scores. We consider
        # words as vertices of the co-occurrency graph. A degree of a word is
        # therefore the sum of all its co-occurrencies.
        word_degrees     = self.co_occurrency_matrix.sum(axis=1)
        word_frequencies = self._compute_word_frequencies()
        word_scores      = self._compute_word_scores(word_degrees, word_frequencies)

        keyword_scores = defaultdict(float)
        for keyword in candidate_keywords:
            score = 0
            for word in keyword:
                score += word_scores[word.lower()]
            keyword_scores[' '.join(keyword)] = score

    def _compute_word_frequencies(self):
        return {word : self.co_occurrency_matrix[word][word] for word in self.all_words}

    def _compute_word_scores(self, word_degrees, word_frequencies):
        word_scores = defaultdict(float)
        for word in self.all_words:
            word_scores[word] = word_degrees[word] / word_frequencies[word]

        return word_scores
