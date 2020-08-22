# SynNLI 

## Usage
- pip install -r requirements
- download NLI data set to data
    - and specify in config
- download allennlp dependency parser and SRL labeler from path
    - and specify in config
- python do_parse
    - parse original jsonl to jsonl with dependencies and srl
- python allennlp train ""
