from pathlib import Path
from typing import Iterable, Tuple

import nltk
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab

from qa_utils.lightning.datasets import PairwiseTrainDatasetBase, PointwiseTrainDatasetBase, ValTestDatasetBase


DMNInput = Tuple[torch.LongTensor, torch.LongTensor, torch.IntTensor]
DMNBatch = Tuple[torch.LongTensor, torch.IntTensor, torch.LongTensor, torch.LongTensor]
DMNPointwiseTrainInput = Tuple[DMNInput, int]
DMNPointwiseTrainBatch = Tuple[DMNBatch, torch.FloatTensor]
DMNPairwiseTrainInput = Tuple[DMNInput, DMNInput]
DMNPairwiseTrainBatch = Tuple[DMNBatch, DMNBatch]
DMNValTestInput = Tuple[int, int, DMNInput, int]
DMNValTestBatch = Tuple[torch.IntTensor, torch.IntTensor, DMNBatch, torch.IntTensor]


def _get_single_dmn_input(query: str, doc: str, vocab: Vocab) -> DMNInput:
    """Tokenize a single (query, document) pair and compute sentence lengths.

    Args:
        query (str): The query
        doc (str): The document
        vocab (Vocab): The vocabulary

    Returns:
        DMNInput: Query tokens, document tokens and sentence lengths
    """
    query_tokens = [vocab.stoi[w] for w in nltk.word_tokenize(query.lower())]
    doc_tokens = []
    sentence_lengths = []
    for sentence in nltk.sent_tokenize(doc.lower()):
        sentence_tokens = [vocab.stoi[w] for w in nltk.word_tokenize(sentence)]
        doc_tokens.extend(sentence_tokens)
        sentence_lengths.append(len(sentence_tokens))
    return torch.LongTensor(query_tokens), \
           torch.LongTensor(doc_tokens), \
           torch.IntTensor(sentence_lengths)


def _collate_dmn(inputs: Iterable[DMNInput], pad_id: int) -> DMNBatch:
    """Collate a number of DMN inputs.

    Args:
        inputs (Iterable[DMNInput]): The inputs
        pad_id (int): The padding value

    Returns:
        DMNBatch: Query tokens, query lengths, document tokens, sentence lengths
    """
    batch_query_tokens, batch_doc_tokens, batch_sentence_lengths = zip(*inputs)
    query_lengths = [len(x) for x in batch_query_tokens]
    return pad_sequence(batch_query_tokens, batch_first=True, padding_value=pad_id), \
           torch.IntTensor(query_lengths), \
           pad_sequence(batch_doc_tokens, batch_first=True, padding_value=pad_id), \
           pad_sequence(batch_sentence_lengths, batch_first=True, padding_value=0)


class DMNPointwiseTrainDataset(PointwiseTrainDatasetBase):
    """Dataset for pointwise DMN training.

    Args:
        data_file (Path): Data file containing queries and documents
        train_file (Path): Trainingset file
        vocab (Vocab): Vocabulary
    """
    def __init__(self, data_file: Path, train_file: Path, vocab: Vocab):
        super().__init__(data_file, train_file)
        self.pad_id = vocab.stoi['<pad>']
        self.vocab = vocab

    def get_single_input(self, query: str, doc: str) -> DMNInput:
        """Create a single model input from a query and a document.

        Args:
            query (str): The query
            doc (str): The document

        Returns:
            DMNInput: The model input
        """
        return _get_single_dmn_input(query, doc, self.vocab)

    def collate_fn(self, train_inputs: Iterable[DMNPointwiseTrainInput]) -> DMNPointwiseTrainBatch:
        """Collate a number of pointwise inputs.

        Args:
            train_inputs (Iterable[DMNPointwiseTrainInput]): The inputs

        Returns:
            DMNPointwiseTrainBatch: A batch of pointwise inputs
        """
        inputs, labels = zip(*train_inputs)
        return _collate_dmn(inputs, self.pad_id), torch.FloatTensor(labels)


class DMNPairwiseTrainDataset(PairwiseTrainDatasetBase):
    """Dataset for pairwise DMN training.

    Args:
        data_file (Path): Data file containing queries and documents
        train_file (Path): Trainingset file
        vocab (Vocab): Vocabulary
    """
    def __init__(self, data_file: Path, train_file: Path, vocab: Vocab):
        super().__init__(data_file, train_file)
        self.pad_id = vocab.stoi['<pad>']
        self.vocab = vocab

    def get_single_input(self, query: str, doc: str) -> DMNInput:
        """Create a single model input from a query and a document.

        Args:
            query (str): The query
            doc (str): The document

        Returns:
            DMNInput: The model input
        """
        return _get_single_dmn_input(query, doc, self.vocab)

    def collate_fn(self, inputs: Iterable[DMNPairwiseTrainInput]) -> DMNPairwiseTrainBatch:
        """Collate a number of pairwise inputs.

        Args:
            inputs (Iterable[DMNPairwiseTrainInput]): The inputs

        Returns:
            DMNPairwiseTrainBatch: A batch of pairwise inputs
        """
        pos_inputs, neg_inputs = zip(*inputs)
        return _collate_dmn(pos_inputs, self.pad_id), _collate_dmn(neg_inputs, self.pad_id)


class DMNValTestDataset(ValTestDatasetBase):
    """Dataset for DMN validation/testing.

    Args:
        data_file (Path): Data file containing queries and documents
        val_test_file (Path): Validationset/testset file
        vocab (Vocab): Vocabulary
    """
    def __init__(self, data_file: Path, val_test_file: Path, vocab: Vocab):
        super().__init__(data_file, val_test_file)
        self.pad_id = vocab.stoi['<pad>']
        self.vocab = vocab

    def get_single_input(self, query: str, doc: str) -> DMNInput:
        """Create a single model input from a query and a document.

        Args:
            query (str): The query
            doc (str): The document

        Returns:
            DMNInput: The model input
        """
        return _get_single_dmn_input(query, doc, self.vocab)

    def collate_fn(self, val_test_inputs: Iterable[DMNValTestInput]) -> DMNValTestBatch:
        """Collate a number of validation/testing inputs.

        Args:
            val_test_inputs (Iterable[DMNValTestInput]): The inputs

        Returns:
            DMNValTestBatch: A batch of validation inputs
        """
        q_ids, doc_ids, inputs, labels = zip(*val_test_inputs)
        return torch.IntTensor(q_ids), \
               torch.IntTensor(doc_ids), \
               _collate_dmn(inputs, self.pad_id), \
               torch.IntTensor(labels)
