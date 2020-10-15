import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import _get_anchor_negative_triplet_mask, _get_anchor_positive_triplet_mask, _get_triplet_mask, \
    batch_hard_triplet_loss, batch_all_triplet_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction=self.reduction)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class GBFG(nn.Module):
    def __init__(self, delta=1,  device='cpu', reduction='mean'):
        super(GBFG, self).__init__()
        self.delta = delta
        self.reduction = reduction
        self.device = device

    def forward(self, inputs, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')

        grad = torch.autograd.grad(
            outputs=ce_loss,
            inputs=inputs,
            grad_outputs=torch.ones_like(ce_loss).to(self.device),
            create_graph=True
        )[0]
        min_idx = torch.argmin(ce_loss)

        grad_min = grad[min_idx.item()].view(-1, 1).unsqueeze(dim=0)
        grad_min = grad_min.expand(grad.size(0), grad_min.size(1), 1)  # Broad-cast grad_min batch-wise

        forget_loss = self.delta * torch.bmm(grad.view(grad.size(0), grad.size(1), -1), grad_min).reshape(-1)
        loss = ce_loss - forget_loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class FGFL(nn.Module):
    def __init__(self, gamma=2, delta=1, n_classes=2, device='cpu', reduction='mean'):
        super(FGFL, self).__init__()
        self.gamma = gamma
        self.delta = delta
        self.reduction = reduction

        self.device = device
        self.c = torch.zeros(n_classes).to(self.device)

    def forward(self, inputs, outputs, targets):
        uq, counts = torch.unique(targets, return_counts=True)
        self.c[uq] += counts
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')

        grad = torch.autograd.grad(
            outputs=ce_loss,
            inputs=inputs,
            grad_outputs=torch.ones_like(ce_loss).to(self.device),
            create_graph=True
        )[0]
        min_idx = torch.argmin(ce_loss)
        pt = torch.exp(-ce_loss)
        arrive_rate = (self.c[targets]/torch.sum(self.c))
        focal_loss = arrive_rate * (1 - arrive_rate * pt) ** self.gamma * ce_loss

        grad_min = grad[min_idx.item()].view(-1, 1).unsqueeze(dim=0)
        grad_min = grad_min.expand(grad.size(0), grad_min.size(1), 1)  # Broad-cast grad_min batch-wise

        forget_loss = torch.bmm(grad.view(grad.size(0), grad.size(1), -1), grad_min).reshape(-1)
        focal_loss -= self.delta * forget_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class OTFL(nn.Module):
    def __init__(self, alpha=2.0, device='cpu', reduction='mean'):
        super(OTFL, self).__init__()
        self.alpha = alpha
        self.device = device
        self.reduction = reduction

    def forward(self, x, px, y):
        uq = torch.unique(y).cpu().numpy()
        ce_x = F.cross_entropy(px, y.long(), reduction='none')
        # Compute grad wrt the current batch
        grad_x = torch.autograd.grad(
            outputs=ce_x,
            inputs=x,
            grad_outputs=torch.ones_like(ce_x).to(self.device),
            create_graph=True
        )[0]

        triplet_fg_loss = 0.0
        for m in uq:
            mask = y == m
            mask_neg = y != m
            ce_m = ce_x[mask]
            if ce_m.size(0) != 0:
                # Select anchor and hard positive instances for class m
                positive_batch = x[mask]
                anchor_idx = torch.argmin(ce_m)
                anchor = positive_batch[anchor_idx].unsqueeze(dim=0)
                grad_a = grad_x[mask][anchor_idx]
                # anchor should not equal positive
                positive_batch = torch.cat((positive_batch[:anchor_idx], positive_batch[anchor_idx + 1:]), dim=0)

                anchor_batch = anchor.expand(positive_batch.size())  # Broad-cast grad_min batch-wise
                positive_dist = F.pairwise_distance(anchor_batch.view(anchor_batch.size(0), -1),
                                                    positive_batch.view(positive_batch.size(0), -1))
                hard_positive_idx = torch.argmax(positive_dist)
                grad_p = grad_x[mask][hard_positive_idx]
                # Select hard negative instances
                negative_batch = x[mask_neg]
                anchor_batch = anchor.expand(negative_batch.size())  # Broad-cast grad_min batch-wise
                negative_dist = F.pairwise_distance(anchor_batch.view(anchor_batch.size(0), -1),
                                                    negative_batch.view(negative_batch.size(0), -1))
                hard_negative_idx = torch.argmin(negative_dist)
                grad_n = grad_x[mask_neg][hard_negative_idx]

                triplet_fg_loss += F.cosine_similarity(grad_a.view(-1), grad_p.view(-1), dim=0) \
                                   - F.cosine_similarity(grad_a.view(-1), grad_n.view(-1), dim=0)

        triplet_fg_loss /= len(uq)
        loss = ce_x - self.alpha * triplet_fg_loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

