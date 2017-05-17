# -*- coding: utf-8 -*-

"""Module for Tokenizer interface."""

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Tokenizer interface. Forces a common structure among tokenizers."""

    @staticmethod
    @abstractmethod
    def tokenize_sentences(document_path):
        pass

    @classmethod
    @abstractmethod
    def tokenize_keywords(cls, texts, max_size=3, flexible_window=False):
        """Extract candidate keywords from one or more text sources.

        Args:
            texts (iterable): An iterable of strings. For example, a list of
                sentences: ['first sentence', 'second sentence', ...]
            max_size (int): The maximum number of words that each keyword will
                be made of.
            flexible_window (bool): if True, gather all ngrams whose length goes
                from 1 up to the maximum size provided. If False, only return
                the longer ngram available and discard shorter ones.

        Yields:
            A generator of keywords. Each keyword is a string composed of up
            to 3 word tokens.

        """
        pass
