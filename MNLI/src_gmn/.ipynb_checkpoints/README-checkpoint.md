# SynNLI 

## Usage (Cur)
- ./install_dependencies.sh 
- download NLI style data set to data
    - and specify in "./src_gmn/training_config.jsonnet"
- download allennlp dependency parser and SRL labeler from path
    - and specify in "./src_gmn/training_config.jsonnet"
- parse data (see Parse Data section)
- allennlp train "./src_gmn/training_config.jsonnet" -s "./param/testv1"   --include-package "package_v1" --force

## Parse Data

## Future Supported Usage
- pip install -r requirements
