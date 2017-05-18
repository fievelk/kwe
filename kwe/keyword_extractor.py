#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Keyword extraction module for KWE."""

import itertools
from collections import defaultdict

import pandas as pd

from kwe.tokenizers import RegexpKeywordTokenizer

class KeywordExtractor(object):
    """Keyword extraction class.

    Attributes:
        input_file (str): The path leading to the text file from which
            keywords should be extracted.
        max_keyword_size (int): The maximum number of word tokens that can
            compose a single keyword.
        co_occurrency_matrix (pandas.DataFrame): A matrix of co-occurrencies
            between words. A co-occurrency occurs when two words belong to the
            same keyword candidate.
        all_words (set): The set of all word tokens in the target text.
        tokenizer (class): The tokenizer class used to extract sentences and
            word tokens from the target text.

    """

    def __init__(self, input_file, max_keyword_size=3, tokenizer=None):
        """Constructor for the KeywordExtractor class.

        Args:
            input_file (str): The path leading to the text file from which
                keywords should be extracted.
            max_keyword_size (int): [optional] The maximum number of word tokens
                for each keyword. Default: 3.
            tokenizer (class): [optional] A tokenizer class implementing the
                Tokenizer interface. Default: RegexpKeywordTokenizer.

        """
        self.input_file = input_file
        self.max_keyword_size = max_keyword_size
        self.co_occurrency_matrix = None
        self.all_words = None

        if tokenizer is None:
            self.tokenizer = RegexpKeywordTokenizer
        else:
            self.tokenizer = tokenizer

    def extract_keywords(self):
        """Extract keywords from the provided input file.

        Returns:
            A list of keywords.

        """

        candidate_keywords = self._extract_candidate_keywords()
        self._build_co_occurrency_matrix(candidate_keywords)
        keyword_scores = self._compute_all_keyword_scores(candidate_keywords)
        pruned_keyword_scores = self._prune_candidate_keywords(keyword_scores)

        return [keyword for keyword, score in keyword_scores.items()]

    def _extract_candidate_keywords(self):
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

            # Lists are unhashable; we therefore use tuples so that we can
            # have keywords as dictionary keys
            keyword_scores[tuple(keyword)] = score

        return keyword_scores

    def _compute_word_frequencies(self):
        return {word : self.co_occurrency_matrix[word][word] for word in self.all_words}

    def _compute_word_scores(self, word_degrees, word_frequencies):
        word_scores = defaultdict(float)
        for word in self.all_words:
            word_scores[word] = word_degrees[word] / word_frequencies[word]

        return word_scores

    def _prune_candidate_keywords(self, keyword_scores):
        """Return best `n` keyword candidates. `n` is computed as one-third the
        number of words in the co-occurrency graph.
        See Mihalcea and Tarau (2004).

        """
        n = len(self.all_words) // 3

        return sorted(
            keyword_scores.items(),
            key=lambda x: x[1],
            reverse=True)[:n]
