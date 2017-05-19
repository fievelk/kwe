#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Keyword extraction module for KWE.

The module follows an hybrid approach, combining an intra-document keyword
scoring (RAKE approach) and an inter-document comparison using a corpus of
text files.

Sources:
    - Rose, Stuart, et al.
        "Automatic keyword extraction from individual documents."
        Text Mining (2010): 1-20.
    - Mihalcea, Rada, and Paul Tarau.
        "TextRank: Bringing order into texts."
        Association for Computational Linguistics, 2004.

"""

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
        co_occurrence_matrix (pandas.DataFrame): A matrix of co-occurrencies
            between words. A co-occurrence occurs when two words belong to the
            same candidate keyword.
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
        self.co_occurrence_matrix = None
        self.all_words = None

        if tokenizer is None:
            self.tokenizer = RegexpKeywordTokenizer
        else:
            self.tokenizer = tokenizer

    def extract_keywords(self):
        """Extract keywords from the provided input file.

        Returns:
            list: A list of keywords.

        """
        candidate_keywords = self._extract_candidate_keywords()
        self._build_co_occurrence_matrix(candidate_keywords)
        keyword_scores = self._compute_all_keyword_scores(candidate_keywords)
        pruned_keyword_scores = self._prune_candidate_keywords(keyword_scores)

        return [keyword for keyword, score in keyword_scores.items()]

    def _extract_candidate_keywords(self):
        """Extract all possible keywords without pruning the final result.
        These candidate keywords can be subsequently sorted and pruned to
        achieve meaningful results.

        Returns:
            list: A list of candidate keywords. Each keyword is a list of word tokens.

        """
        sentences = self.tokenizer.tokenize_sentences(self.input_file)
        return list(self.tokenizer.tokenize_keywords(
            sentences, self.max_keyword_size
        ))

    def _build_co_occurrence_matrix(self, candidate_keywords):
        """Build a matrix of words co-occurrencies. A co-occurrence occurs
        when two words belong to the same candidate keyword.

        Args:
            candidate_keywords (iterable): A list of candidate keywords.
                Each keyword is a list of strings (word tokens).

        Returns:
            pandas.DataFrame: A DataFrame object representing a matrix of word
                co-occurrencies.

        Notes:
            A word co-occurring with itself represents an increment of
            the word frequency. Word frequencies are therefore aligned over
            the matrix diagonal.

        """
        # Populate the set of all word types included in candidate keywords
        self.all_words = {
            word.lower()
            for word in itertools.chain.from_iterable(candidate_keywords)
        }

        # Create the co-occurrence matrix
        self.co_occurrence_matrix = pd.DataFrame(
            0, index=self.all_words, columns=self.all_words
        )

        # Populate the co-occurrence matrix with words co-occurrencies.
        for kw in candidate_keywords:
            for w1 in kw:
                for w2 in kw:
                    self.co_occurrence_matrix[w1.lower()][w2.lower()] += 1

    def _compute_all_keyword_scores(self, candidate_keywords):
        """Compute intra-document keyword scores based on the tecnique described
        in Rose, Stuart, et al. (2010). See specific methods for more details.

        We consider words as vertices of the co-occurrence graph.
        - The degree of a word is the sum of all its co-occurrencies.
        - The frequency of a word is the number of times the word occurs among
        all candidate keywords.
        - The score of a word is the ratio between the word degree and the word
        frequency (deg/freq).

        The final keyword score is the sum of the scores of its words.

        Args:
            candidate_keywords (iterable): A list of candidate keywords.
                Each keyword is a list of strings (word tokens).

        Returns:
            defaultdict (str: float): A dictionary with keywords as keys and
            word scores as values. Each keyword is a string.

        """
        # Create a pandas Series containing word degree scores.
        word_degrees     = self.co_occurrence_matrix.sum(axis=1)
        word_frequencies = self._compute_word_frequencies()
        word_scores      = self._compute_word_scores(word_degrees, word_frequencies)

        keyword_scores = defaultdict(float)
        for keyword in candidate_keywords:
            score = 0
            for word in keyword:
                score += word_scores[word.lower()]

            # Lists are unhashable and tuples cannot be directly handled by
            # gemsim; we therefore concatenate keyword tokens so that we can
            # have keywords as dictionary keys
            keyword_scores[' '.join(keyword)] = score

        return keyword_scores

    def _compute_word_frequencies(self):
        """Compute all word frequencies.

        Word frequencies can be obtained by traversing the co-occurrence matrix
        diagonal.

        Returns:
            dict (str: int): A dictionary whose keys are word tokens and
                whose values are frequency counts.

        """
        return {word : self.co_occurrence_matrix[word][word] for word in self.all_words}

    def _compute_word_scores(self, word_degrees, word_frequencies):
        """Compute all word scores.

        Args:
            word_degrees (pandas.Series): a dictionary-like object of word
                degrees. Word tokens should be used as keys to retrieve their
                relative degree.
            word_frequencies (dict): a dictionary whose keys are word tokens and
                whose values are frequency counts.

        Returns:
            defaultdict (str: float): A dictionary whose keys are word tokens
                and whose values are word scores.

        """

        word_scores = defaultdict(float)
        for word in self.all_words:
            word_scores[word] = word_degrees[word] / word_frequencies[word]

        return word_scores

    def _prune_candidate_keywords(self, keyword_scores):
        """Return best `n` candidate keywords. `n` is computed as one-third the
        number of words in the co-occurrence graph.
        See Mihalcea and Tarau (2004).

        Args:
            keyword_scores (dict): A dictionary whose keys are keywords and
            whose values are word scores.

        Returns:
            list: A list of (keyword, score) tuples sorted by score (in
                decreasing order).

        """
        n = len(self.all_words) // 3

        return sorted(
            keyword_scores.items(),
            key=lambda x: x[1],
            reverse=True)[:n]
