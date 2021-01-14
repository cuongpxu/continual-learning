import torch
import math
from torch.optim.optimizer import Optimizer


class AdLR(Optimizer):
    r"""Implements the algorithm proposed in the paper "A novel adaptive learning rate algorithm for CNN training"
    Author: S.V. Georgakopoulos and V.P. Plagianakos
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (required)
        gamma1: meta-learning rate 1
        gamma2: meta-learning rate 2
        gamma3: meta-learning rate 3
    """

    def __init__(self, params, lr=0.01, gamma1=0.01, gamma2=0.001, gamma3=0.01, device='cpu'):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, gamma1=gamma1, gamma2=gamma2, gamma3=gamma3)
        super(AdLR, self).__init__(params, defaults)
        self.init_lr = lr
        self.device = device

        # Initialize previous grad
        for k, p in enumerate(self.param_groups[0]['params']):
            if k == 0:
                self.prev_grad1 = torch.zeros(p.view(-1).size()).to(device)
            else:
                self.prev_grad1 = torch.cat((self.prev_grad1, torch.zeros(p.view(-1).size()).to(self.device)))

        self.prev_grad2 = self.prev_grad1.detach().clone().to(self.device)

    def __setstate__(self, state):
        super(AdLR, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for k, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad = p.grad
                if k == 0:
                    all_grad = torch.flatten(p.grad)
                else:
                    all_grad = torch.cat((all_grad, torch.flatten(p.grad)))

                # Update weight
                p.add_(grad, alpha=-group['lr'])

            # Update learning rate
            group['lr'] = group['lr'] + \
                              group['gamma1'] * torch.dot(self.prev_grad1, all_grad.to(self.device)) + \
                              group['gamma2'] * torch.dot(self.prev_grad2, self.prev_grad1)
            if group['lr'] < 0:
                group['lr'] = group['gamma3'] * self.init_lr

            # Store prev grad
            self.prev_grad2 = self.prev_grad1.detach().clone().to(self.device)
            self.prev_grad1 = all_grad.detach().clone().to(self.device)

        return loss


class eAdLR(Optimizer):
    r"""Implements the algorithm proposed in the paper "Efficient learning rate adaptation for CNN training"
    Author: S.V. Georgakopoulos and V.P. Plagianakos
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (required)
        gamma1: meta-learning rate 1
        gamma2: meta-learning rate 2
        gamma3: meta-learning rate 3
    """

    def __init__(self, params, lr=0.01, gamma1=0.01, gamma2=0.001, gamma3=0.01, device='cpu'):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, gamma1=gamma1, gamma2=gamma2, gamma3=gamma3)
        super(eAdLR, self).__init__(params, defaults)
        self.init_lr = lr
        self.iter = 0
        self.device = device

        # Initialize previous grad
        self.prev_grad1 = dict({})
        self.prev_grad2 = dict({})
        for k, p in enumerate(self.param_groups[0]['params']):
            self.prev_grad1[k] = torch.zeros(p.size()).to(self.device)
            self.prev_grad2[k] = torch.zeros(p.size()).to(self.device)

    def __setstate__(self, state):
        super(eAdLR, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for k, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad

                # Update weight
                p.add_(grad, alpha=-group['lr'])

                # Update learning rate
                if self.iter >= 3:
                    group['lr'] = group['lr'] + \
                                  group['gamma1'] * torch.dot(torch.flatten(self.prev_grad1[k]), torch.flatten(grad).to(self.device)) +\
                                  group['gamma2'] * torch.dot(torch.flatten(self.prev_grad2[k]), torch.flatten(self.prev_grad1[k]))
                    if group['lr'] < 0:
                        group['lr'] = group['gamma3'] * self.init_lr

                # Store prev grad
                self.prev_grad2[k] = self.prev_grad1[k].detach().clone().to(self.device)
                self.prev_grad1[k] = grad.detach().clone().to(self.device)

        self.iter += 1
        return loss


class AdamAdLR(Optimizer):
    r"""Integrating the algorithm proposed in the paper "Efficient learning rate adaptation for CNN training"
    Author: S.V. Georgakopoulos and V.P. Plagianakos with ADAM optimizer
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (required)
        gamma1: meta-learning rate 1
        gamma2: meta-learning rate 2
        gamma3: meta-learning rate 3
    """

    def __init__(self, params, lr=0.01, gamma1=0.01, gamma2=0.001, gamma3=0.01,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, device='cpu'):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, gamma1=gamma1, gamma2=gamma2, gamma3=gamma3, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamAdLR, self).__init__(params, defaults)
        self.init_lr = lr
        self.iter = 0
        self.device = device

        # Initialize previous grad
        self.prev_grad1 = dict({})
        self.prev_grad2 = dict({})
        for k, p in enumerate(self.param_groups[0]['params']):
            self.prev_grad1[k] = torch.zeros(p.size()).to(self.device)
            self.prev_grad2[k] = torch.zeros(p.size()).to(self.device)

    def __setstate__(self, state):
        super(AdamAdLR, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for k, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad

                # Update weight
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format).to(self.device)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format).to(self.device)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format).to(self.device)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Update learning rate
                if self.iter >= 3:
                    group['lr'] = group['lr'] + \
                                  group['gamma1'] * torch.dot(torch.flatten(self.prev_grad1[k]), torch.flatten(grad).to(self.device)) +\
                                  group['gamma2'] * torch.dot(torch.flatten(self.prev_grad2[k]), torch.flatten(self.prev_grad1[k]))
                    if group['lr'] < 0:
                        group['lr'] = group['gamma3'] * self.init_lr

                # Store prev grad
                self.prev_grad2[k] = self.prev_grad1[k].detach().clone().to(self.device)
                self.prev_grad1[k] = grad.detach().clone().to(self.device)

        self.iter += 1
        return loss
