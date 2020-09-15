# Uploading to PyPI

## reference
- https://packaging.python.org/tutorials/packaging-projects/

## algorithm
- create `setup.py`, `REANDME.md`, `license.txt`, `test`, etc.
- python3 -m pip install --user --upgrade setuptools wheel
- python3 setup.py sdist bdist_wheel
- python3 -m pip install --user --upgrade twine
- python3 -m twine upload dist/*
- now the project is available on https://pypi.org/project/syn-nli/0.0.1/