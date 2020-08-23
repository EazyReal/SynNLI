import torch
from typing import Dict

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


def sparse2dense(data, batch_indices):
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




##################################################
# for sparse cross attention
##################################################

def compute_cross_attention(x, y, sim):
    """Compute cross attention.
    
    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    y_j attend to x_i:
    a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))
    attention_x = sum_j a_{i->j} y_j
    attention_y = sum_i a_{j->i} x_i

    Args:
    x: NxD float tensor.
    y: MxD float tensor.
    sim: a (x, y) -> similarity function.

    Returns:
    attention_x: NxD float tensor.
    attention_y: MxD float tensor.
    """
    a = sim(x, y)
    a_x = torch.nn.softmax(a, axis=1)  # i->j
    a_y = torch.nn.softmax(a, axis=0)  # j->i
    attention_x = torch.matmul(a_x, y)
    attention_y = torch.matmul(a_y.transpose(-1,-2), x)
    return attention_x, attention_y


def batch_block_pair_attention(data,
                               block_idx,
                               n_blocks,
                               similarity='dotproduct'):
    """Compute batched attention between pairs of blocks.

    This function partitions the batch data into blocks according to block_idx.
    For each pair of blocks, x = data[block_idx == 2i], and
    y = data[block_idx == 2i+1], we compute

    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    y_j attend to x_i:
    a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))

    and

    attention_x = sum_j a_{i->j} y_j
    attention_y = sum_i a_{j->i} x_i.

    Args:
    data: NxD float tensor.
    block_idx: N-dim int tensor.
    n_blocks: integer.
    similarity: a string, the similarity metric.

    Returns:
    attention_output: NxD float tensor, each x_i replaced by attention_x_i.

    Raises:
    ValueError: if n_blocks is not an integer or not a multiple of 2.
    """
    if not isinstance(n_blocks, int):
        raise ValueError('n_blocks (%s) has to be an integer.' % str(n_blocks))

    if n_blocks % 2 != 0:
        raise ValueError('n_blocks (%d) must be a multiple of 2.' % n_blocks)

    sim = get_pairwise_similarity(similarity)

    results = []

    # This is probably better than doing boolean_mask for each i
    partitions = tf.dynamic_partition(data, block_idx, n_blocks)

    # It is rather complicated to allow n_blocks be a tf tensor and do this in a
    # dynamic loop, and probably unnecessary to do so.  Therefore we are
    # restricting n_blocks to be a integer constant here and using the plain for
    # loop.
    for i in range(0, n_blocks, 2):
        x = partitions[i]
        y = partitions[i + 1]
        attention_x, attention_y = compute_cross_attention(x, y, sim)
        results.append(attention_x)
        results.append(attention_y)

    results = tf.concat(results, axis=0)
    # the shape of the first dimension is lost after concat, reset it back
    results.set_shape(data.shape)
    return results
