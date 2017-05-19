# KWE

KWE is a KeyWord Extraction tool written in Python

## Requirements

- `python 3`
- `gensim`
- `nltk`
- `pandas`

Optional dependencies for testing:

- `tox`
- `pytest`

## Installation

```bash
git clone git@github.com:fievelk/kwe.git
cd kwe/
pip install .
```

## Usage

```python
from kwe import KeywordExtractor

# Gather all file paths for our target document and corpus documents
target_file_path = 'kwe/data/script.txt'
corpus_file_paths = [
    'kwe/data/transcript_1.txt',
    'kwe/data/transcript_2.txt',
    'kwe/data/transcript_3.txt'
]

# Retrieve top n (`limit`) keyword candidates.
all_keywords = KeywordExtractor.extract_corpus_keywords(
    target_file_path, corpus_file_paths, max_keyword_size=3, limit=10)
```

## Testing

To run tests in several Python environments simply issue the `tox` command from terminal. Please check `tox.ini` for more details.
