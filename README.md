# suggestion

Requirements:

* Anaconda Python 3.5+ (other distributions may work too)
* Working C compiler
* Yelp academic dataset...
* node.js

Setup:

    (git clone https://github.com/kcarnold/kenlm && cd kenlm && pip install .)
    ./setup.sh
    pip install -e .


Preprocessing:

    python scripts/preprocess_yelp.py
    ./scripts/make_model.sh yelp_train

Run backend:

    ./run

Setup frontend:
    cd frontend
    npm install

Run frontend:

    cd frontend
    npm start
