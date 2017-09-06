#!/bin/bash
conda install -y tornado ujson cython numpy datrie nltk scipy numba scikit-learn cytoolz pandas
pip install -r requirements.txt
pip install -e .

# Dev install:
# conda install -y fabric3 tqdm
