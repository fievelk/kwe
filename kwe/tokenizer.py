# -*- coding: utf-8 -*-

"""Module for tokenization components and classes."""

from itertools import groupby

from nltk.tokenize import sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import stopwords

class KeywordTokenizer(object):
    """Tokenizer class for keyword extraction."""

    @staticmethod
    def tokenize_sentences(document_path):
        with open(document_path) as document:
            for line in document:
                yield from sent_tokenize(line)

    @classmethod
    def tokenize_keywords(cls, texts, max_size=3):
        """Extract candidate keywords from one or more text sources.

        Args:
            texts (iterable): An iterable of strings. For example, a list of
                sentences: ['first sentence', 'second sentence', ...]
            max_size (int): The maximum number of words that each keyword will
                be made of.

        Yields:
            A generator of keywords. Each keyword is a string composed of up
            to 3 word tokens.

        """
        for text in texts:
            word_tokens = TreebankWordTokenizer().tokenize(text)
            chunks_without_stopwords = list(cls._split_at_stopwords(word_tokens))

            yield cls.extract_ngrams(chunks_without_stopwords, size=max_size)

    @staticmethod
    def extract_ngrams(tokens, size=3):
        """Extract ngrams of up to the specified size from a list of given
        tokens.

        >>> from kwe.tokenizer import KeywordTokenizer
        >>> tokens = [['Food'], ['substance', 'consumed', 'daily', 'hospital']]
        >>> KeywordTokenizer.extract_ngrams(tokens) # doctest: +NORMALIZE_WHITESPACE
        [['Food'], ['substance'], ['consumed'], ['daily'], ['hospital'],
        ['substance', 'consumed'], ['consumed', 'daily'], ['daily', 'hospital'],
        ['substance', 'consumed', 'daily'], ['consumed', 'daily', 'hospital']]

        Args:
            tokens (iterable): An iterable containing lists of words. E.g.:
                [['Food'], ['substance', 'consumed'], ...]
            size (int): The ngram size. E.g.: 2 for bigrams, 3 for trigrams, etc.

        Yields:
            A list of lists. Each sublist represents the ngrams of up to `size`
            words extracted from a single token. E.g.:
            [['Food'], ['substance'], ['consumed'], ['substance', 'consumed'], ...]

        """
        keyword_tokens = []
        for word_tokens in tokens:
            for n in range(1,max(size+1, len(word_tokens))):
                for i in range(len(word_tokens)-n+1):
                    keyword_tokens.append(word_tokens[i:i+n])

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
        stopwords_set = set(stopwords.words('english'))
        for k, g in groupby(word_tokens, key=lambda w: w.lower() in stopwords_set):
            if not k:
                yield list(g)
