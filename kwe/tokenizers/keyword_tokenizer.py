# -*- coding: utf-8 -*-

"""Module for tokenization components and classes."""

from itertools import groupby
from string import punctuation

from nltk.tokenize import sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import stopwords

from .tokenizer import Tokenizer

punctuation_list = set(punctuation)
additional_punctuation = {'–', '°'}
punctuation_list.update(additional_punctuation)

stopwords_set = set(stopwords.words('english'))

class KeywordTokenizer(Tokenizer):
    """Tokenizer class for keyword extraction."""

    @staticmethod
    def tokenize_sentences(document_path):
        with open(document_path) as document:
            for line in document:
                yield from sent_tokenize(line)

    @classmethod
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

        Examples:
            >>> from kwe.tokenizers import KeywordTokenizer
            >>> sents = [
            ...     'Wolves are an endangered species',
            ...     'Food is any substance consumed to provide nutritional support for the body.'
            ... ]
            >>> list(KeywordTokenizer.tokenize_keywords(sents, 3, True)) # doctest: +NORMALIZE_WHITESPACE
            [['Wolves'], ['endangered'], ['species'], ['endangered', 'species'],
            ['Food'], ['substance'], ['consumed'], ['substance', 'consumed'],
            ['provide'], ['nutritional'], ['support'], ['provide', 'nutritional'],
            ['nutritional', 'support'], ['provide', 'nutritional', 'support'],
            ['body']]
            >>> list(KeywordTokenizer.tokenize_keywords(sents, 3, False)) # doctest: +NORMALIZE_WHITESPACE
            [['Wolves'], ['endangered', 'species'], ['Food'], ['substance',
            'consumed'], ['provide', 'nutritional', 'support'], ['body']]

        """
        tokenizer = TreebankWordTokenizer()

        for text in texts:
            # Preprocessing steps
            word_tokens = tokenizer.tokenize(text)

            word_tokens = cls.remove_punctuation(word_tokens)
            chunks_without_stopwords = list(cls._split_at_stopwords(word_tokens))

            yield from cls.extract_ngrams(
                tokens=chunks_without_stopwords,
                size=max_size,
                flexible_window=flexible_window
            )

    @staticmethod
    def remove_punctuation(tokens):
        """Remove puctuation tokens from an iterable of tokens.

        Args:
            tokens (iterable): An iterable of strings. E.g. a list
                of words and symbols in a sentence:
                ['Wolves', ':', 'an', 'endangered', 'species', '.']

        Returns:
            A list of word tokens without punctuation elements.

        Examples:
            >>> from kwe.tokenizers import KeywordTokenizer
            >>> tokens = ['Wolves', ':', 'an', 'endangered', 'species', '.']
            >>> KeywordTokenizer.remove_punctuation(tokens)
            ['Wolves', 'an', 'endangered', 'species']

        """
        return [tok for tok in tokens if tok not in punctuation_list]

    @staticmethod
    def extract_ngrams(tokens, size=3, flexible_window=False):
        """Extract ngrams of up to the specified size from a list of given
        tokens.

        Args:
            tokens (iterable): An iterable containing lists of words. E.g.:
                [['Food'], ['substance', 'consumed'], ...]
            size (int): The ngram size. E.g.: 2 for bigrams, 3 for trigrams, etc.
            flexible_window (bool): if True, gather all ngrams whose length goes
                from 1 up to the maximum size provided. If False, only return
                the longer ngram available and discard shorter ones.

        Yields:
            A list of lists. Each sublist represents the ngrams of up to `size`
            words extracted from a single token. E.g.:
            [['Food'], ['substance'], ['consumed'], ['substance', 'consumed'], ...]

        Examples:
            >>> from kwe.tokenizers import KeywordTokenizer
            >>> tokens = [['Food'], ['substance', 'consumed', 'daily', 'hospital']]
            >>> KeywordTokenizer.extract_ngrams(tokens, flexible_window=True) # doctest: +NORMALIZE_WHITESPACE
            [['Food'], ['substance'], ['consumed'], ['daily'], ['hospital'],
            ['substance', 'consumed'], ['consumed', 'daily'], ['daily', 'hospital'],
            ['substance', 'consumed', 'daily'], ['consumed', 'daily', 'hospital']]
            >>> KeywordTokenizer.extract_ngrams(tokens, flexible_window=False) # doctest: +NORMALIZE_WHITESPACE
            [['Food'], ['substance', 'consumed', 'daily'], ['consumed', 'daily', 'hospital']]
            >>> tokens = [['a', 'b', 'c', 'd', 'e']]
            >>> KeywordTokenizer.extract_ngrams(tokens, 2, flexible_window=True) # doctest: +NORMALIZE_WHITESPACE
            [['a'], ['b'], ['c'], ['d'], ['e'], ['a', 'b'], ['b', 'c'], ['c', 'd'],
            ['d', 'e']]

        """
        assert all(isinstance(tok, list) for tok in tokens), 'tokens should be a list of lists'

        keyword_tokens = []

        if flexible_window:
            for word_tokens in tokens:
                for n in range(1, min(size, len(word_tokens))+1):
                    for i in range(len(word_tokens)-n+1):
                        keyword_tokens.append(word_tokens[i:i+n])
        else:
            for word_tokens in tokens:
                max_size = min(size, len(word_tokens))
                for i in range(len(word_tokens)-max_size+1):
                    keyword_tokens.append(word_tokens[i:i+max_size])

        return keyword_tokens

    def _split_at_stopwords(word_tokens):
        """Split an iterable of words into chunks, using English stopwords as
        separators.

        Args:
            word_tokens (iterable): An iterable of strings. E.g. a list
                of words in a sentence:
                ['Wolves', 'are', 'an', 'endangered', 'species']

        Yields:
            A list of lists. Each sublist is made of words. E.g.:

        """
        for k, g in groupby(word_tokens, key=lambda w: w.lower() in stopwords_set):
            if not k:
                yield list(g)
