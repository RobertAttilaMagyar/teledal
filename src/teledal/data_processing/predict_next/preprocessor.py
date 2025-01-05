import argparse
import re
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from .._special_constants import (
    EMBEDDING_DIM,
    END_OF_SEQUENCE,
    PADDING_VALUE,
    START_OF_SEQUENCE,
)


class Preprocessor:
    def __init__(
        self,
        regex_pairs: list[tuple[str, str]] = [],
        device: str = None,
        hidden_state_idx: int = -2,
    ):
        """
        Preprocessor for TeleDAL

        Parameters:
        -----------
        - regex_pairs: The regex substitutions required for the textual preprocessing of the
        log lines
        """

        self._device = (
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
            if device is None
            else device
        )
        self._tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased"
        )
        self._vocab_keys = list(self._tokenizer.vocab.keys())
        self._bert_model = BertModel.from_pretrained(
            "bert-base-uncased", output_hidden_states=True
        )
        self._bert_model.to(self._device)
        self._bert_model.eval()

        self._hidden_state_idx = hidden_state_idx

        self._embedding_dim = EMBEDDING_DIM

        self._regex_pairs = regex_pairs

        self.sequence_length = None

    def regexize(self, msg: str) -> str:
        for pair in self._regex_pairs:
            msg = re.sub(pair[0], pair[1], msg)

        return msg

    def encode_message(self, msg: str, regex: bool = True) -> torch.Tensor:
        """
        Creates semantic embedding of a log message by
        embedding each token and summing up the vectors along
        the first axis.

        Parameters:
        -----------
        - msg: The log message to embed.
        - regex: Decides whether or not to apply regex on msg.
        """
        if regex:
            msg = self.regexize(msg)
        max_len = 100
        encode_dict = self._tokenizer.batch_encode_plus(
            [msg],  # Sentence to encode
            add_special_tokens=True,  # Add [CLS] and [SEP] tokens
            max_length=max_len,  # Pad & truncate sentences
            truncation=True,  # Truncate if needed
            padding=True,  # Pad to max length
            return_attention_mask=True,  # Create an attention mask that ignores paddings
            return_tensors="pt",  # Return type should be torch.Tensor
        )

        with torch.no_grad():
            hidden_states = self._bert_model(
                encode_dict["input_ids"].to(self._device), encode_dict["attention_mask"]
            )[2][self._hidden_state_idx]

        embedding = hidden_states.sum(dim=1)[0, :]
        return embedding

    @staticmethod
    def pad_to_length(embeddings: list[torch.Tensor], length: int = None):
        assert embeddings, "Empty list passed as embeddings!"
        embedding_dim = embeddings[0].size()[0]
        device = embeddings[0].device

        padding_length = None
        if length:
            embeddings = embeddings[:length]
            padding_length = length - len(embeddings)

        start_of_sequence = torch.ones(embedding_dim) * START_OF_SEQUENCE
        start_of_sequence = start_of_sequence.to(device)
        end_of_sequence = torch.ones(embedding_dim) * END_OF_SEQUENCE
        end_of_sequence = end_of_sequence.to(device)

        result = torch.vstack([start_of_sequence] + embeddings + [end_of_sequence])

        return F.pad(
            result, (0, 0, 0, (padding_length if length else 0)), value=PADDING_VALUE
        )

    def encode_sequence(
        self, log_sequence: list[str], length: int = None
    ) -> torch.Tensor:
        """
        Encodes a whole log sequence and adds a special start of sequence and
        end of sequence vector embedding.

        Parameters:
        -----------
        log_sequence: List of log messages in the given sequence.
        length: Maximum length of a log sequence.
        """

        msg_embeddings = []
        for msg in log_sequence:
            embedding = self.encode_message(msg)
            msg_embeddings.append(embedding)

        return self.pad_to_length(msg_embeddings, length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to raw text files with the names of sequence identifiers containing the raw formats of the log lines",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to write the preprocessed log sequence embeddings",
    )
    parser.add_argument(
        "--regex-pairs",
        type=Path,
        default=None,
        help="Path to a `.yaml` file containing the regex pairs",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="Extract log sequences with this length",
    )

    args = parser.parse_args()

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    args.output_dir.mkdir(parents=True)

    regex_pairs: list[tuple[str, str]] = []

    if args.regex_pairs is not None:
        with open(args.regex_pairs) as f:
            regex_pairs = [
                (re.compile(v), f" {k} ") for k, v in yaml.safe_load(f).items()
            ]

    pp = Preprocessor(regex_pairs)

    files = [file for file in args.input_dir.glob("*")]
    for file in tqdm(files):
        with open(file) as f:
            sequence = f.readlines()

        embedding = pp.encode_sequence(sequence, args.sequence_length)

        out_file = (args.output_dir / file.stem).with_suffix(".pt")

        torch.save(embedding, out_file)
