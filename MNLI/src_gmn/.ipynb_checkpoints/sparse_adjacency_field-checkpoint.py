from typing import Dict, List, Set, Tuple
import logging
import textwrap

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data.fields.field import Field, TextField
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.vocabulary import Vocabulary

from torch_geometric.data.data import Data as PyGeoData

logger = logging.getLogger(__name__)


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
        we do not construct sequencefield from node_attr, since need tokenizer 
    label_namespace : `str`, optional (default=`'labels'`)
        The namespace to use for converting tag strings into integers.  We convert tag strings to
        integers for you, and this parameter tells the `Vocabulary` object which mapping from
        strings to integers to use (so that "O" as a tag doesn't get the same id as "O" as a word).
    # we do not need padding here, may need follow_batch (but in this implementation, we treat node_attr to follow_batch)
    """

    __slots__ = [
        "indices",
        "labels",
        "sequence_field",
        "_label_namespace",
        "_padding_value",
        "_indexed_labels",
    ]

    # It is possible that users want to use this field with a namespace which uses OOV/PAD tokens.
    # This warning will be repeated for every instantiation of this class (i.e for every data
    # instance), spewing a lot of warnings so this class variable is used to only log a single
    # warning per namespace.
    _already_warned_namespaces: Set[str] = set()

    def __init__(
        self,
        graph: PyGeoData,
        sequence_field: SequenceField,
        label_namespace: str = "labels"
    ) -> None:
        labels = graph.edge_attr : Union[List[str], List[int]] # label can skip indexing
        self.indices = graph.edge_index : torch.tensor
        self.labels = labels
        self.sequence_field = sequence_field
        self._label_namespace = label_namespace
        self._indexed_labels: List[int] = None

        self._maybe_warn_for_namespace(label_namespace)
        field_length = sequence_field.sequence_length()
        
        # we do not check duplicate edge, since edge can be multi label
        #if len(set(indices)) != len(indices):
        #    raise ConfigurationError(f"Indices must be unique, but found {indices}")
        
        # check for out-of-index edge
        if not all(
            0 <= index[1] < field_length and 0 <= index[0] < field_length for index in indices
        ):
            raise ConfigurationError(
                f"Label indices and sequence length "
                f"are incompatible: {indices} and {field_length}"
            )
        
        # if labels is passed, should have same length with edges 
        if labels is not None and len(indices) != len(labels):
            raise ConfigurationError(
                f"Labelled indices were passed, but their lengths do not match: "
                f" {labels}, {indices}"
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
        return {"num_tokens": self.sequence_field.sequence_length()}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        desired_num_tokens = padding_lengths["num_tokens"]
        tensor = torch.ones(desired_num_tokens, desired_num_tokens) * self._padding_value
        labels = self._indexed_labels or [1 for _ in range(len(self.indices))]

        for index, label in zip(self.indices, labels):
            tensor[index] = label
        return tensor

    @overrides
    def empty_field(self) -> "AdjacencyField":

        # The empty_list here is needed for mypy
        empty_list: List[Tuple[int, int]] = []
        adjacency_field = AdjacencyField(
            empty_list, self.sequence_field.empty_field(), padding_value=self._padding_value
        )
        return adjacency_field

    def __str__(self) -> str:
        length = self.sequence_field.sequence_length()
        formatted_labels = "".join(
            "\t\t" + labels + "\n" for labels in textwrap.wrap(repr(self.labels), 100)
        )
        formatted_indices = "".join(
            "\t\t" + index + "\n" for index in textwrap.wrap(repr(self.indices), 100)
        )
        return (
            f"AdjacencyField of length {length}\n"
            f"\t\twith indices:\n {formatted_indices}\n"
            f"\t\tand labels:\n {formatted_labels} \t\tin namespace: '{self._label_namespace}'."
        )

    def __len__(self):
        return len(self.sequence_field)