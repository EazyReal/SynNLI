import torch
from typing import Dict, List

# actually do not get the sorted advantage here
# but ok for experimental model
def sorted_dynamic_parition(x: torch.Tensor, b: torch.Tensor) -> List[torch.Tensor]:
    N = b[-1] + 1
    res = []
    for i in range(N):
        res += [x[(b == i).nonzero(as_tuple=False).squeeze(1)]]
    return res

def dense2sparse(data: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    usage: convert dense batch(for non-GNN) to sparse batch (for GNN)
    a gather_nd method for pytorch with dimension = 3, 2
    
    input size = (B, batchN, D), (B,batchN)
    ouput size = (allN, D), (allN)
    
    example:
    
     {'data': tensor([[[0.7879, 0.0682, 0.6570, 0.7031, 0.8994],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
 
         [[0.8266, 0.3092, 0.3913, 0.4549, 0.1064],
          [0.8491, 0.6805, 0.2992, 0.1845, 0.9280],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
 
         [[0.3792, 0.4077, 0.3438, 0.0266, 0.2067],
          [0.4379, 0.9779, 0.2881, 0.4637, 0.7458],
          [0.9262, 0.2924, 0.1877, 0.8627, 0.9728],
          [0.6972, 0.2883, 0.9224, 0.2346, 0.8337]]]),
     'mask': tensor([[ True, False, False, False],
             [ True,  True, False, False],
             [ True,  True,  True,  True]])}
             
    tensor([[0.7879, 0.0682, 0.6570, 0.7031, 0.8994],
        [0.8266, 0.3092, 0.3913, 0.4549, 0.1064],
        [0.8491, 0.6805, 0.2992, 0.1845, 0.9280],
        [0.3792, 0.4077, 0.3438, 0.0266, 0.2067],
        [0.4379, 0.9779, 0.2881, 0.4637, 0.7458],
        [0.9262, 0.2924, 0.1877, 0.8627, 0.9728],
        [0.6972, 0.2883, 0.9224, 0.2346, 0.8337]])
    tensor([0, 1, 1, 2, 2, 2, 2])
    """
    b, n, d = data.size()
    indices = mask.nonzero(as_tuple=False)
    batch_ids = indices.T[0]
    out = torch.stack([data[tuple(idx)] for idx in indices])
    return {"data": out, "batch_indices": batch_ids}


def sparse2dense(data: torch.Tensor, batch_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    usage: convert sparse batch (for GNN) to dense batch(for non-GNN)
    a scatter_nd method for pytorch with dimension = 3, 2
    besure to remenber properties of original tensor
    also besure that the data and batch_indices is valid
    
    todo: check backpro and memory issue
    
    input = 
        data (N, d), batch_indices (N)
    output =
        data (b, n, d), mask = (b, n)
        
    example:
    tensor([[0.7879, 0.0682, 0.6570, 0.7031, 0.8994],
        [0.8266, 0.3092, 0.3913, 0.4549, 0.1064],
        [0.8491, 0.6805, 0.2992, 0.1845, 0.9280],
        [0.3792, 0.4077, 0.3438, 0.0266, 0.2067],
        [0.4379, 0.9779, 0.2881, 0.4637, 0.7458],
        [0.9262, 0.2924, 0.1877, 0.8627, 0.9728],
        [0.6972, 0.2883, 0.9224, 0.2346, 0.8337]])
    tensor([0, 1, 1, 2, 2, 2, 2])
    
    
    {'data': tensor([[[0.7879, 0.0682, 0.6570, 0.7031, 0.8994],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
 
         [[0.8266, 0.3092, 0.3913, 0.4549, 0.1064],
          [0.8491, 0.6805, 0.2992, 0.1845, 0.9280],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
 
         [[0.3792, 0.4077, 0.3438, 0.0266, 0.2067],
          [0.4379, 0.9779, 0.2881, 0.4637, 0.7458],
          [0.9262, 0.2924, 0.1877, 0.8627, 0.9728],
          [0.6972, 0.2883, 0.9224, 0.2346, 0.8337]]]),
     'mask': tensor([[ True, False, False, False],
             [ True,  True, False, False],
             [ True,  True,  True,  True]])}
    """
    _, seqlens = torch.unique_consecutive(batch_indices, return_counts=True)
    # size
    b = batch_indices[-1] + 1
    n = torch.max(seqlens)
    d = data.size()[1]
    # tensor info
    dtype=data.dtype
    device=data.device
    requires_grad=data.requires_grad
    #
    indices_y = torch.cat([ torch.tensor(list(range(l))) for l in seqlens ], dim=0)
    indices_y = indices_y.to(device=device)
    indices = torch.cat([batch_indices.unsqueeze(dim=1), indices_y.unsqueeze(dim=1)], dim=1).T
    true_tensor = torch.Tensor([True]*indices.size()[1]).to(device=device)
    out = torch.sparse.FloatTensor(indices, data).to_dense()
    mask = torch.sparse.LongTensor(indices, true_tensor).to_dense().to(dtype=bool)
    return {"data": out, "mask": mask}

