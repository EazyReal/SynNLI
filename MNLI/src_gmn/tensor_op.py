import torch
from typing import Dict

def dense2sparse(input_: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    usage: convert dense batch(for non-GNN) to sparse batch (for GNN)
    a gather_nd method for pytorch with dimension = 3, 2
    
    input size = (B, batchN, D), (B,batchN)
    ouput size = (allN, D), (allN)
    """
    b, n, d = input_.size()
    #out = torch.masked_select(input_.permute(2,0,1), mask)
    #out = out.view(-1, d).contiguous()
    # this implementation will get incorrect output with ok shape
    indices = mask.nonzero()
    batch_ids = indices.T[0] # this part is batch ids
    #out = input_.select(indices)
    out = torch.stack([input_[tuple(idx)] for idx in indices])
    return {"data": out, "batch_indices": batch_ids}


def sparse2dense(input_, batch_indices):
    """
    usage: convert sparse batch (for GNN) to dense batch(for non-GNN)
    a scatter_nd method for pytorch with dimension = 3, 2
    besure to remenber properties of original tensor
    also besure that the input_ and batch_indices is valid
    
    todo: check backpro and memory issue
    
    input = 
        input_ (N, d), batch_indices (N)
    output =
        data (b, n, d), mask = (b, n)
    """
    _, seqlens = torch.unique_consecutive(batch_indices, return_counts=True)
    b = batch_indices[-1] + 1 # index + 1
    n = torch.max(seqlens)
    d = input_.size()[1]
    indices_y = torch.cat([ torch.tensor(list(range(l))) for l in seqlens ], dim=0)
    indices = torch.cat([batch_indices.unsqueeze(dim=1), indices_y.unsqueeze(dim=1)], dim=1)
    mask = torch.full([b, n], False, dtype=bool)
    out = torch.zeros([b, n, d], dtype=input_.dtype, device=input_.device, requires_grad=input_.requires_grad)
    for i, index in enumerate(indices):
        tid = tuple(index)
        out[tid] = input_[i] # i not tid
        mask[tid] = True
    return {"data": out, "mask": mask}