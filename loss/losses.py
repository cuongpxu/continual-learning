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
    def __init__(self, n_dim=10, n_classes=2, alpha=2.0, margin=0.05, var='all', device='cpu', reduction='mean'):
        super(OTFL, self).__init__()
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.alpha = alpha
        self.margin = margin
        self.var = var
        self.device = device
        self.reduction = reduction

        self.anchors = torch.zeros((n_classes, 1, n_dim), requires_grad=True).to(self.device)

    def get_anchor_batch(self, grads, targets):
        anchor_batch = torch.zeros(targets.size(0), self.n_dim).to(self.device)
        for i in range(targets.size(0)):
            anchor_batch[i] = grads[targets[i]].data.clone()
        anchor_batch.requires_grad_(True)
        return anchor_batch

    def get_negative_batch(self, grads, targets):
        uq = torch.unique(targets, return_counts=False)
        negative_batch = torch.zeros(grads.size(0), self.n_dim).to(self.device)
        for i in range(grads.size(0)):
            diff = uq[uq != targets[i]]
            if len(diff) != 0:
                if self.var == 'ran':
                    # Select negative sample randomly
                    diff_class = diff[np.random.randint(len(diff))]
                    for j in range(grads.size(0)):
                        if targets[j] == diff_class:
                            negative_batch[i] = grads[j].data.clone()
                            break
                elif self.var == 'all':
                    # Compute with all negative samples in the batch
                    for j in range(grads.size(0)):
                        if targets[j] in diff:
                            negative_batch[i] += grads[j].squeeze().data.clone()
                elif self.var == 'min':
                    # Compute with selected negative samples in terms of some constraints
                    res = []
                    index = []
                    with torch.no_grad():
                        for j in range(grads.size(0)):
                            if targets[j] in diff:
                                res.append(torch.dot(grads[i].squeeze(), grads[j].squeeze()))
                                index.append(j)
                    negative_batch[i] += grads[index[np.argmin(res)]].squeeze().data.clone()
                else:
                    raise NotImplementedError('Not implemented {} version of OTFL'.format(self.var))
        negative_batch.requires_grad_(True)
        return negative_batch

    def forward(self, x, px, pa, y):
        ce_x = F.cross_entropy(px, y, reduction='none')
        # Compute grad wrt the current batch
        grad_x = torch.autograd.grad(
            outputs=ce_x,
            inputs=x,
            grad_outputs=torch.ones_like(ce_x).to(self.device),
            create_graph=True
        )[0]

        # Compute grad wrt the anchors
        ce_a = F.cross_entropy(pa, torch.arange(self.n_classes).long().to(self.device), reduction='none')
        grad_a = torch.autograd.grad(
            outputs=ce_a,
            inputs=self.anchors,
            grad_outputs=torch.ones_like(ce_a).to(self.device),
            create_graph=True
        )[0]
        # print(grad_x.size(), grad_a.size())
        anchor_batch = self.get_anchor_batch(grad_a, y)
        negative_batch = self.get_negative_batch(grad_x.view(grad_x.size(0), grad_x.size(1), -1), y)

        anchor_positive = torch.bmm(grad_x.view(grad_x.size(0), grad_x.size(1), -1), anchor_batch.unsqueeze(dim=2))
        anchor_negative = torch.bmm(negative_batch.unsqueeze(dim=1), anchor_batch.unsqueeze(dim=2))

        triplet_fg_loss = anchor_negative.squeeze() - anchor_positive.squeeze() + self.margin

        loss = ce_x + self.alpha * triplet_fg_loss

        # Store gradient of the last min loss instance of every class label
        # min_idx = torch.argmin(ce_x)
        # self.grad_anchors[y[min_idx.item()]] = grad_x[min_idx.item()].data.clone()
        # Store min loss instance for each class label
        for m in range(self.n_classes):
            mask = y == m
            ce_m = ce_x[mask]
            if ce_m.size(0) != 0:
                min_m = torch.min(ce_m)
                idx = ce_x == min_m
                self.anchors[m] = x[idx][0].view(-1).unsqueeze(dim=0).data.clone()

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

