from typing import Union, Dict, List
from pathlib import Path

import torch 
import numpy as np
import matplotlib.pyplot as plt

def show_sequence_attention(seq: List[str], att: Union[torch.Tensor, np.array], serialization_dir: Union[str, Path]=None) -> None:
    """
    attention visualization
    seq is List[List[Token]]
    att is {"data": torch.Tensor, "mask": torch.BoolTensor}
    """
    # Set up figure with colorbar
    fig = plt.figure() # figsize=None
    ax = fig.add_subplot(111)
    # data = att["data"][0][att["mask"][0] ==True].T
    # if use raw batch att return
    # print(data.shape)
    cax = ax.imshow(att, cmap='bone', origin='upper')
    fig.colorbar(cax)
    # Set up axes
    ax.set_xticks(np.arange(len(seq)))
    ax.set_xticklabels(seq)
    ax.set_yticklabels([])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


    # Show label at every tick
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title("Attention")
    fig.tight_layout()
    if serialization_dir is not None:
        plt.savefig(serialization_dir)
    plt.show()
    
def show_matrix_attention(seq1: List[str], seq2: List[str], att: Union[torch.Tensor, np.array], serialization_dir: Union[str, Path]=None)->None:
    """
    attention visualization
    seq1/seq2 is List[Token]
    att is 
    """
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(att, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticks(np.arange(len(seq1)))
    ax.set_yticks(np.arange(len(seq2)))
    ax.set_xticklabels(seq1, rotation=90)
    ax.set_yticklabels(seq2)

    # Show label at every tick
    #import matplotlib.ticker as ticker
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title("Attention")
    fig.tight_layout()
    if serialization_dir is not None:
        plt.savefig(serialization_dir)
    plt.show()