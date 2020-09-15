from typing import Dict, List, Set, Tuple, Union, TypeVar
import logging
import textwrap

from collections import defaultdict
from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data.fields import (Field, TextField, SequenceField)
#from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.vocabulary import Vocabulary

from torch_geometric.data.data import Data as PyGeoData
from torch_geometric.data.batch import Batch as PyGeoBatch

DataArray = TypeVar(
    "DataArray", torch.Tensor, Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]
)

logger = logging.getLogger(__name__)

# for input model
SparseAdjacencyFieldTensors = Dict[str, torch.Tensor]

# author = ytlin 
class SparseAdjacencyField(Field[torch.Tensor]):
    """
    A `SparseAdjacencyField` defines directed adjacency relations between elements
    in a :class:`~allennlp.data.fields.sequence_field.SequenceField`.
    Because it's a labeling of some other field, we take that field as input here
    and use it to determine our padding and other things.
    This field will handle the batching and vocaburary indexing of pytorch geometric Data type.
    
    # Parameters
    graph: `PyGeoData`
        Note that in PyGeoData
        edge_attr is list of edge attributes (List[Any])
        node_attr is list of node attributes (List[Any])
        edge_index is a [2, |V|] tensor
        # can be a way to implement this w/o using PyGeoData
    sequence_field: `SequenceField`
        I do not construct sequencefield from node_attr, since need tokenizer 
    label_namespace : `str`, optional (default=`'labels'`)
        The namespace to use for converting tag strings into integers.  We convert tag strings to
        integers for you, and this parameter tells the `Vocabulary` object which mapping from
        strings to integers to use (so that "O" as a tag doesn't get the same id as "O" as a word).
    #  do not need padding here, may need follow_batch (but in this implementation, we treat node_attr to follow_batch)
    
    """
    __slots__ = (
        "edge_indices",
        "labels",
        "sequence_field",
        "_label_namespace",
        "_indexed_labels",
    )
    

    # It is possible that users want to use this field with a namespace which uses OOV/PAD tokens.
    # This warning will be repeated for every instantiation of this class (i.e for every data
    # instance), spewing a lot of warnings so this class variable is used to only log a single
    # warning per namespace.
    _already_warned_namespaces: Set[str] = set()

    def __init__(
        self,
        graph: PyGeoData,
        sequence_field: SequenceField,
        label_namespace: str = "edge_labels"
    ) -> None:
        # Field inheritor dose not need to super().__init__() 
        # label skip index is not implemented yet, todo
        labels: Union[List[str], List[int]] = graph.edge_attr
        self.edge_indices = graph.edge_index
        self.labels = labels
        self.sequence_field = sequence_field
        self._label_namespace = label_namespace
        self._indexed_labels: List[int] = None

        self._maybe_warn_for_namespace(label_namespace)
        field_length = sequence_field.sequence_length()
        
        # we do not check duplicate edge, since edge can be multi label
        #if len(set(edge_indices)) != len(edge_indices):
        #    raise ConfigurationError(f"edge_indices must be unique, but found {edge_indices}")
        
        if self.edge_indices is not None: # do not check for empty_field
            # check for out-of-index edge
            if not all(
                0 <= self.edge_indices[1][i] < field_length and 0 <= self.edge_indices[0][i] < field_length for i in range(self.edge_indices.size()[1])
            ):
                raise ConfigurationError(
                    f"Label edge_indices and sequence length "
                    f"are incompatible: {self.edge_indices} and {field_length}"
                )

            # if labels is passed, should have same length with edges 
            if labels is not None and self.edge_indices.size()[1] != len(labels):
                raise ConfigurationError(
                    f"Labelled edge_indices were passed, but their lengths do not match: "
                    f" {labels}, {self.edge_indices}"
                )

    def _maybe_warn_for_namespace(self, label_namespace: str) -> None:
        if not (self._label_namespace.endswith("labels") or self._label_namespace.endswith("tags")):
            if label_namespace not in self._already_warned_namespaces:
                logger.warning(
                    "Your label namespace was '%s'. We recommend you use a namespace "
                    "ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by "
                    "default to your vocabulary.  See documentation for "
                    "`non_padded_namespaces` parameter in Vocabulary.",
                    self._label_namespace,
                )
                self._already_warned_namespaces.add(label_namespace)

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if self._indexed_labels is None and self.labels is not None:
            for label in self.labels:
                counter[self._label_namespace][label] += 1  # type: ignore

    @overrides
    def index(self, vocab: Vocabulary):
        if self.labels is not None:
            self._indexed_labels = [
                vocab.get_token_index(label, self._label_namespace) for label in self.labels
            ]
    
    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        """
        If there are things in this field that need padding, note them here. 
        In order to pad a batch of instance, we get all of the lengths from the batch, take the max, and pad
        everything to that length (or use a pre-specified maximum length).
        The return value is a dictionary mapping keys to lengths, like `{'num_tokens': 13}`.
        This is always called after :func:`index`.
        """
        return {"num_tokens": len(self.sequence_field)}
    
    # this is the main difference between Sparse and Dense Adjacency
    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        tensor_dict = {}
        tensor_dict["edge_attr"] = torch.tensor(self._indexed_labels, dtype=torch.long)
        tensor_dict["edge_index"] = self.edge_indices
        tensor_dict["batch_id"] = torch.zeros(len(self.sequence_field)) # batch_id for the node_attr
        return tensor_dict
    
    # use pytorch geometric sparse graph batching style
    # I looked up the references for coding this part
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/batch.html#Batch.from_data_list
    # https://github.com/allenai/allennlp/blob/master/allennlp/data/fields/text_field.py
    @overrides 
    def batch_tensors(self, tensor_list: List[DataArray]) -> DataArray:  # type: ignore
        """
        Takes the output of `Field.as_tensor()` from a list of `Instances` and merges it into
        one batched tensor for this `Field`. 
        
        Input: List[TensorDict]
        Output: TensorDict
        """
        pre_sum = 0
        batch_tensor = defaultdict(list)
        
        for batch_id, tensor in enumerate(tensor_list):
            cur_edge_index = tensor["edge_index"].add(pre_sum)
            cur_nodes = len(tensor["batch_id"])
            batch_tensor["edge_index"].append(cur_edge_index)
            batch_tensor["edge_attr"].append(tensor["edge_attr"])
            cur_id_tensor = torch.full(size=[cur_nodes], fill_value=batch_id, dtype=torch.long) # as a 1*d tensor
            batch_tensor["batch_id"].append(cur_id_tensor)
            pre_sum += cur_nodes
        batch_tensor["edge_index"] = torch.cat(batch_tensor["edge_index"], dim=1)
        batch_tensor["edge_attr"] = torch.cat(batch_tensor["edge_attr"], dim=0)
        batch_tensor["batch_id"] = torch.cat(batch_tensor["batch_id"], dim=0)
        return dict(batch_tensor)
    
    # tested that can build, not know whats for, may be errors in downstream
    @overrides
    def empty_field(self) -> "SparseAdjacencyField":

        # The empty_list here is needed for mypy
        empty_adjacency_field = SparseAdjacencyField(
            graph=PyGeoData(), sequence_field=self.sequence_field.empty_field()
        )
        return empty_adjacency_field

    def __str__(self) -> str:
        length = self.sequence_field.sequence_length()
        formatted_labels = "".join(
            "\t\t" + labels + "\n" for labels in textwrap.wrap(repr(self.labels), 100)
        )
        formatted_edge_indices = "".join(
            "\t\t" + index + "\n" for index in textwrap.wrap(repr(self.edge_indices), 100)
        )
        return (
            f"AdjacencyField of length {length}\n"
            f"\t\twith edge_indices:\n {formatted_edge_indices}\n"
            f"\t\tand labels:\n {formatted_labels} \t\tin namespace: '{self._label_namespace}'."
        )

    def __len__(self):
        return len(self.sequence_field)