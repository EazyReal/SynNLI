# versions
CUDA="cu101"
TORCH="1.6.0"
ALLENNLP="1.1.0rc3"
TORCH_VISION="0.7.0"

# allennlp related
pip install allennlp==${ALLENNLP}
pip install allennlp-models==${ALLENNLP}

# pytorch geometric related
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-geometric

pip install torch-scatter
pip install torch-sparse
pip install torch-cluster
pip install torch-spline-conv

# pytorch related
pip install torch==${TORCH}+${CUDA} torchvision==${TORCH_VISION}+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html

# stanza to be added