import re
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F


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
        self._bert_model.eval()

        self._hidden_state_idx = hidden_state_idx

        self._embedding_dim = 768

        self._regex_pairs = regex_pairs

    def encode_message(self, msg: str) -> torch.Tensor:
        """
        Creates semantic embedding of a log message by
        embedding each token and summing up the vectors along
        the first axis.

        Parameters:
        -----------
        - msg: The log message to embed.
        """
        for pair in self._regex_pairs:
            msg = re.sub(pair[0], pair[1], msg)
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
                encode_dict["input_ids"], encode_dict["attention_mask"]
            )[2][self._hidden_state_idx]

        embedding = hidden_states.sum(dim=1)[0, :]
        return embedding

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
        padding_length = None
        if length: 
            log_sequence = log_sequence[:length]
            padding_length = length - len(log_sequence)
        
        start_of_sequence = torch.zeros(self._embedding_dim)
        end_of_sequence = torch.ones(self._embedding_dim)/self._embedding_dim


        msg_embeddings = []
        for msg in log_sequence:
            embedding = self.encode_message(msg)
            msg_embeddings.append(embedding)

        result = torch.vstack([start_of_sequence] + msg_embeddings + [end_of_sequence])
    
        return F.pad(result, (0,0,0,(padding_length if length else 0)), value = -1)
        
    