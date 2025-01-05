from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from teledal.data_processing.predict_next.data import PredictNextData


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
    pass


class LayerNormLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.recurrent_list = nn.ModuleList()
        self.norm_list = nn.ModuleList()
        for i in range(num_layers):
            self.recurrent_list.append(
                nn.LSTM(
                    input_size=(input_size if i == 0 else hidden_size),
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
            )
            self.norm_list.append(nn.LayerNorm(normalized_shape=hidden_size, eps=eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        h0 = h0.to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = c0.to(x.device)
        out = x
        for i in range(self.num_layers):
            out, _ = self.recurrent_list[i](out, (h0[[i], ...], c0[[i], ...]))
            out = self.norm_list[i](out)

        return out


class TeleDAL(nn.Module):
    def __init__(
        self,
        # seq_length: int,
        k: int = 5,
        embedding_dim: int = 768,
        hidden_dim: int = 32,
        num_layers: int = 2,
        bidirectional: bool = False,
    ) -> "TeleDAL":
        super().__init__()
        self.k = k
        self.embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self.custom_lstm = LayerNormLSTM(embedding_dim, hidden_dim, num_layers)
        self.topk = nn.Linear(hidden_dim, embedding_dim * k)

    def forward(self, x: torch.Tensor):  # x in shape (N, seq_length + 2, embedding_dim)
        out = self.custom_lstm(x)
        out = self.topk(out)
        out = out.view(-1, x.size(1), self.k, self.embedding_dim)

        return out  # (N, seq_length, K, embedding_dim)


if __name__ == "__main__":
    ds = PredictNextData(
        Path(
            "/home/ad.adasworks.com/attila.magyar/Desktop/Thesis/teledal/data/HDFS_embeddings"
        )
    )
    loader = DataLoader(ds, 3)
    model = TeleDAL()
    for data in loader:
        X, y, _ = data
        print(X.shape)
        out = model(X)
        print(out.shape)
        break
