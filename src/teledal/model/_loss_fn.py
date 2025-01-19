import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from teledal.data_processing._special_constants import PADDING_VALUE


class _LossGolden(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        """
        A less effective but easier to understand implementation of the top k loss
        function serving as the golden code for the custom loss implementation
        """
        super().__init__(size_average, reduce, reduction)

    @staticmethod
    def single_line_loss(y: torch.Tensor, Z: torch.Tensor):
        # y is a vector of dimensions E
        # Z is a matrix with dimensions k x E
        mask = y != PADDING_VALUE
        y = y * mask
        Z = Z * torch.vstack([mask] * Z.shape[0])
        u = F.cosine_similarity(y, Z, dim=1)
        w = (Z.T * u).sum(dim=1)
        ell = F.cosine_similarity(y, w, dim=0)

        return ell.item()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        # y_true: NxLxE
        # y_pred: NxLxkXE
        N, L = y_true.shape[:2]
        mx = torch.zeros(N, L)
        for i in range(N):
            for j in range(L):
                mx[i, j] = self.single_line_loss(y_true[i, j, :], y_pred[i, j, ...])

        mx = 1 + mx
        mx, _ = mx.max(dim=-1)
        if self.reduction == "mean":
            return mx.mean()


class TopKLoss(_Loss):
    """
    Custom loss function for TeleDAL.

    Parameters:
    -----------
    `y_true`: Tensor of shape N x L x E
    `y_pred`: Tensor of shape N x L x k x E
    """

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true = y_true.unsqueeze(-2)
        mask = y_true != PADDING_VALUE
        y_true = y_true * mask
        y_pred = y_pred * mask.expand(y_pred.shape)

        u = F.cosine_similarity(y_true, y_pred, dim=-1).unsqueeze(-1)
        w = (y_pred.mul(u)).sum(dim=-2)
        ell = F.cosine_similarity(y_true.squeeze(), w, dim=-1)
        ell = 1 + ell
        ell, _ = ell.max(dim=-1)

        if self.reduction == "mean":
            return ell.mean()


if __name__ == "__main__":
    generator = torch.Generator().manual_seed(42)
    y_true = torch.randn((2, 3, 10), generator=generator)
    y_pred = torch.randn((2, 3, 5, 10), generator=generator)

    y_true[1, 1, 4:] = PADDING_VALUE

    golden_criteria = _LossGolden()
    criteria = TopKLoss()
    mx = golden_criteria(y_true, y_pred)
    mx2 = criteria(y_true, y_pred)
    print(mx)
    print(mx2)
