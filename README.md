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
import kwe
kwe.runner.extract_keywords('file_path.txt')
```

## Testing

To run tests in several Python environments simply issue the `tox` command from terminal. Please check `tox.ini` for more details.
