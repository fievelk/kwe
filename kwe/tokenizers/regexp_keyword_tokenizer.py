# -*- coding: utf-8 -*-

"""Module for tokenization using regular expressions."""

from nltk.tokenize import RegexpTokenizer

from .keyword_tokenizer import KeywordTokenizer

class RegexpKeywordTokenizer(KeywordTokenizer):
    """Tokenizer class for keyword extraction using regular expressions."""

    @classmethod
    def tokenize_keywords(cls, texts, max_size=3, flexible_window=False):
        """
        Extract candidate keywords from one or more text sources using
        regular expressions.

        """

        tokenizer = RegexpTokenizer(r"['°\w-]+|[\$\€\£][\d\.]+")
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
