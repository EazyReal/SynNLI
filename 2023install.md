# 2023 failed install methods

## M1 Macbook Max
- pyenv => miniforge3
  - conda cannot create python 3.6 environment 
  
## X86 Ubentu (WSL)
- pyenv => miniforge3 
  - `conda create -n iis36 python 3.6`
  - however, after adding channels (torch, pytorch, allennlp, anaconda, conda-forge)
    - all related to torch or allennlp still fail to install
- pyenv => 3.6.15
  - `pip install -r requirements2.txt`
  - torch_geometry cannot be installed when torch is installed...
    - this may the most promissing path right now...
- pyenv => 3.6.15
  - `install_dependencies.sh` also didn't work