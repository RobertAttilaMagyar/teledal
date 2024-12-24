import torch
import torch.nn.functional as F
import torch.nn as nn
from preprocessor import Preprocessor


class TopK(nn.Module):
    def __init__(self, k: int = 5) -> "TopK":
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


def top_k_loss_one(output: torch.Tensor, target: torch.Tensor):
    """
    Calculates the top-K loss funciton for one sample

    Parameters:
    -----------
    - output: A torch tensor with the shape LxKxE containing the top-K predictions.
    - target: A torch tensor with shape LXE containing the target embedding vector

    In this notation L is the length of the sequence, K is the number of K in top-K and
    E is the dimension of the embedding vectors.
    """

    assert len(target.shape) == 1, "The target should be a 1-dimensional tensor"
    assert len(output.shape) == 2, "Output should be a 2-dimensioal tensor"
    assert output.shape[1] == target.shape[0], "Dimension mismatch"

    u = F.cosine_similarity(output, target, dim=2)
    w = torch.sum(torch.dot(output, F.softmax(u)))
    l = F.cosine_similarity(w, target)
    pass


class TeleDAL(nn.Module):
    def __init__(
        self,
        seq_length: int,
        k: int = 5,
        embedding_dim: int = 768,
        hidden_dim: int = 32,
        num_layers: int = 2,
        bidirectional: bool = False,
    ) -> "TeleDAL":
        super().__init__()
        self._seq_len = seq_length
        self._k = k
        self._layer_input = nn.Linear(embedding_dim, hidden_dim)
        self._lstm = nn.LSTM(
            hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=bidirectional,
        )
        self._fc = nn.Linear(hidden_dim, embedding_dim*k)

    def forward(self, x): # x in shape (N, seq_length, embedding_dim)
        print(x.shape)
        out = self._layer_input(x)
        print(out.shape)
        out, _ = self._lstm(out)
        print(out.shape)
        out = self._fc(out)
        print(out.shape)
        out = out.view((self._seq_len+2, self._k, 768))
        return out


with open(
    "/home/ad.adasworks.com/attila.magyar/Desktop/Thesis/teledal/data/HDFS/blk_-5087939924532795057"
) as f:
    seq = f.readlines()

pp = Preprocessor(device="cpu")
seq = pp.encode_sequence(seq, length = 10)
print("Seqence shape")
print(seq.shape)
model = TeleDAL(10)
out = model(seq)
print(out.shape)