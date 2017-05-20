# Notes

Current results do not seem to be very meaningful. Improvements could be then made to both the structure and the content of the code. The first step to take would be writing tests for all methods currently without coverage, in order to find out whether current problems can be ascribed to implementation errors.

#### Table of Contents:

* [Current approach](#current-approach)
* [Improvements and ideas](#improvements-and-ideas)
* [Data](#data)

## Current approach

I decided to test an approach based on statistical intra-document and inter-document scores for keywords. These are the main steps:

1. Extract candidate keywords on each single document:
   - tokenize input text at sentence level;
   - tokenize each sentence at word level;
   - split results into chunks, using stopwords as delimiters.

2. Prune the list of candidate keywords from (1) and only keep top keywords.

   In order to score each candidate, methods from *Automatic Keyword Extraction from Individual Documents* paper are used.
   NOTE: The approach described in the paper does not use a predefined size for ngrams that compose each keyword. In fact:
   > RAKE uses stop words and phrase delimiters to partition the document text into candidate keywords, which are sequences of content words as they occur in the text. Co-occurrences of words within these candidate keywords are meaningful and allow us to identify word co-occurrence without the application of an arbitrarily sized sliding window. Word associations are thus measured in a manner that automatically adapts to the style and content of the text, enabling adaptive and fine-grained measurement of word co-occurrences that will be used to score candidate keywords.

   The steps involve:

   - creating a co-occurrence matrix of words;
   - computing candidate keywords scores (using word frequency and degree);
   - pruning the list of candidate keywords keeping top `N` candidates. `N` is computed as one-third the number of words in the co-occurrence graph.

3. Compare the input document candidate keywords with all the other documents using TF-IDF scores.

   The final list is then sorted and the top `N` keywods are returned along with their score.


## Improvements and ideas

Here is a list of notes, specific improvements and ideas about further development.

- Use other approaches for the last step: replace TF-IDF with some other method (e.g. unsupervised learning).

- Other approaches could involve parsing text for POS tags and defining specific rules to identify meaningful phrases.

- Try to extract features from text. For example, identifying text headers (e.g. short sentences surrounded by empty lines).

- The `RegexpKeywordTokenizer` subclass might seem quite an overkill. It would be possible to simply pass an NLTK `RegexpTokenizer` instance to the `KeywordTokenizer` class, instead of subclassing it. However, this would be a less flexible solution. In `KeywordExtractor` we are in fact using the `tokenize_sentences()` and `tokenize_keywords()` methods; creating or subclassing an internal implementation of the `Tokenizer` interface can guarantee us that these methods will always be available, regardless of the internal chosen implementation. I therefore prefer using a subclass, because I think that this could provide more flexibility and customizability for further developments.

- `tokenize_sentences()` and `tokenize_keywords()` could be grouped into a single method. We could then just call this method from the `KeywordExtractor` class. Classes implementing the `Tokenizer` interface would have to only implement this method.

- It might be useful to create a CorpusReader class to parse files and return an homogeneous structure. We are currently directly feeding file paths to our classes, but using a corpus reader would provide more flexibility for documents that do not share the same structure.

- Applying a better and more flexible preprocessing step to text would be good. This could involve creating a `Preprocessor` class implementing these steps:
  - removing all "Edit" suffixes from misspelled words ("HistoryEdit", "Pre-modern EuropeEdit", etc.)
  - consider using stemming (but this could be counter-productive)
  - forcing lower case during this step, instead of using it in subsequent steps.


- Creating a `Keyword` class could be useful. For example, it would allow us to use a property to store a "raw" version of the keyword before the preprocessing step. We could then take advantage of less sparse statistics (e.g. using stemming) and yet preserve the original form of the words. Performance issues should be taken into consideration.

- Both lists and tuples are currently used to represent keywords. This is something that should be fixed in order to provide more consistency.

## Data

The provided documents are all related to the same topic (food). This simplifies our task, since we do not have to worry about noise produced by documents of different domains. This is particularly useful during the (last) keyword rating step of our task.
