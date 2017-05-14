# KWE

KWE is a KeyWord Extraction tool written in Python

## Requirements

- `python 3`

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
extractor = KeywordExtractor('file_path.txt', max_keyword_size=3)
keywords = extractor.extract_keywords()
```

## Testing

To run tests in several Python environments simply issue the `tox` command from terminal. Please check `tox.ini` for more details.
