# SynNLI 

## Usage (Cur)
- ./install_dependencies.sh 
- download NLI style data set to data
    - and specify path in jsonne
- parse data (see Parse Data section)
    - and specify path in jsonnet
- train model (see Training Area)
    - with jsonnet

## Parse Data with Stanza
- Stanza will be loaded in preprocess.py
    - the parser version is the one @ 2020/8/22
- use preprocess.py
```
python preprocess.py -i ../data/anli_v1.0/R2/ \
 -o ../data/anli_v1.0_preprocessed/R2/ \
 --files dev.jsonl test.jsonl train.jsonl \
 --force \
 -m 10
```
- if want to use allennlp (less recommended)
    - download allennlp dependency parser and SRL labeler from path

## Training
- refer to "the config.jsonnet"
```
allennlp train "./src_gmn/training_config.jsonnet" -s "./param/testv1"   --include-package "package_v1" --force
```

## Future Supported Usage
- pip install -r requirements
- + add configs folder for various config
- note that should take lemmatized as node attr if use work embedding
- root to spetial token
- use MLP prjection

