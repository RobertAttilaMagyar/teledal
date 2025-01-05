import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class TopKLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean"):
        """
        Custom loss function for TeleDAL.

        Parameters:
        -----------
        `y_true`: Tensor of shape N x L x E
        `y_pred`: Tensor of shape N x L x k x E
        """
        super().__init__(size_average, reduce, reduction)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        u = torch.matmul(y_true.unsqueeze(-2), y_pred.transpose(-2, -1)).squeeze(dim=-2)
        u /= torch.linalg.norm(y_pred, dim=-1)
        u = u / torch.linalg.norm(y_true, dim=-1).unsqueeze(-1)
        u = F.softmax(u, dim=-1)

        w = torch.matmul(y_pred.transpose(-1, -2), u.unsqueeze(-1)).squeeze(-1)
        loss = torch.matmul(w.unsqueeze(-2), y_true.unsqueeze(-1))
        norm_factor1 = torch.linalg.norm(y_true, dim=-1)
        norm_factor1 = norm_factor1.view(*norm_factor1.shape, 1, 1)
        norm_factor2 = torch.linalg.norm(w, dim=-1)
        norm_factor2 = norm_factor2.view(*norm_factor2.shape, 1, 1)
        loss /= norm_factor1 * norm_factor2
        loss, _ = torch.max(loss, dim=1)

        if self.reduction == "mean":
            return loss.mean()

        elif self.reduction == "sum":
            return loss.sum()

        return loss.squeeze()
