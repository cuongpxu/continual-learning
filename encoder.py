import numpy as np
import torch
import torch.nn as nn
import models.resnet as rn
import utils
from torch.nn import functional as F
from linear_nets import MLP, fc_layer
from exemplars import ExemplarHandler
from continual_learner import ContinualLearner
from replayer import Replayer
from torchvision.models import resnet18, ResNet


class Classifier(ContinualLearner, Replayer, ExemplarHandler):
    '''Model for classifying images, "enriched" as "ContinualLearner"-, Replayer- and ExemplarHandler-object.'''

    def __init__(self, image_size, image_channels, classes,
                 fc_layers=3, fc_units=1000, fc_drop=0, fc_bn=False, fc_nl="relu", gated=False,
                 bias=True, excitability=False, excit_buffer=False, binaryCE=False, binaryCE_distill=False, AGEM=False,
                 loss='bce', experiment='splitMNIST'):

        # configurations
        super().__init__()
        self.classes = classes
        self.label = "Classifier"
        self.fc_layers = fc_layers

        # settings for training
        self.binaryCE = binaryCE  # -> use binary (instead of multiclass) prediction error
        self.binaryCE_distill = binaryCE_distill  # -> for classes from previous tasks, use the by the previous model
        #   predicted probs as binary targets (only in Class-IL with binaryCE)
        self.AGEM = AGEM  # -> use gradient of replayed data as inequality constraint for (instead of adding it to)
        #   the gradient of the current data (as in A-GEM, see Chaudry et al., 2019; ICLR)
        self.loss = loss

        # Online mem distilation
        self.is_offline_training = False
        self.is_ready_distill = False
        self.alpha_t = 0.5
        # check whether there is at least 1 fc-layer
        if fc_layers < 1:
            raise ValueError("The classifier needs to have at least 1 fully-connected layer.")

        ######------SPECIFY MODEL------######
        self.experiment = experiment
        if self.experiment in ['CIFAR10', 'CIFAR100']:
            self.fcE = rn.resnet20(classes, pretrained=False)
            self.fcE.linear = nn.Identity()

            self.classifier = fc_layer(64, classes, excit_buffer=True, nl='none', drop=fc_drop)
        elif self.experiment == 'ImageNet':
            ResNet.name = 'ResNet-18'
            self.fcE = resnet18(pretrained=True)
            self.fcE.fc = nn.Identity()

            self.classifier = fc_layer(512, classes, excit_buffer=True, nl='none', drop=fc_drop)
        else:
            # flatten image to 2D-tensor
            self.flatten = utils.Flatten()

            # fully connected hidden layers
            self.fcE = MLP(input_size=image_channels * image_size ** 2, output_size=fc_units, layers=fc_layers - 1,
                           hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias,
                           excitability=excitability, excit_buffer=excit_buffer, gated=gated)
            mlp_output_size = fc_units if fc_layers > 1 else image_channels * image_size ** 2

            # classifier
            self.classifier = fc_layer(mlp_output_size, classes, excit_buffer=True, nl='none', drop=fc_drop)

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.fcE.list_init_layers()
        list += self.classifier.list_init_layers()
        return list

    @property
    def name(self):
        return "{}_c{}".format(self.fcE.name, self.classes)

    def forward(self, x):
        if self.experiment not in ['splitMNIST', 'permMNIST', 'rotMNIST']:
            final_features = self.fcE(x)
        else:
            final_features = self.fcE(self.flatten(x))
        return self.classifier(final_features)

    def feature_extractor(self, images):
        if self.experiment not in ['splitMNIST', 'permMNIST', 'rotMNIST']:
            return self.fcE(images)
        else:
            return self.fcE(self.flatten(images))

    def select_triplets(self, embeds, y_score, x, y, triplet_selection, task, scenario, use_embeddings):
        uq = torch.unique(y).cpu().numpy()
        selection_strategies = triplet_selection.split('-')
        # Select instances in the batch for replay later
        for m in uq:
            mask = y == m
            mask_neg = y != m
            ce_m = y_score[mask]
            if ce_m.size(0) != 0:
                # Select anchor and hard positive instances for class m
                positive_batch = x[mask]
                positive_embed_batch = embeds[mask]
                anchor_idx = torch.argmin(ce_m)
                anchor_x = positive_batch[anchor_idx].unsqueeze(dim=0)
                anchor_embed = positive_embed_batch[anchor_idx].unsqueeze(dim=0)
                # anchor should not equal positive
                positive_batch = torch.cat(
                        (positive_batch[:anchor_idx], positive_batch[anchor_idx + 1:]), dim=0)
                positive_embed_batch = torch.cat(
                    (positive_embed_batch[:anchor_idx], positive_embed_batch[anchor_idx + 1:]), dim=0)
                if positive_batch.size(0) != 0:
                    if use_embeddings:
                        anchor_batch = anchor_embed.expand(positive_embed_batch.size())
                        positive_dist = F.pairwise_distance(anchor_batch.view(anchor_batch.size(0), -1),
                                                            positive_embed_batch.view(positive_embed_batch.size(0), -1))
                    else:
                        anchor_batch = anchor_x.expand(positive_batch.size())
                        positive_dist = F.pairwise_distance(anchor_batch.view(anchor_batch.size(0), -1),
                                                                positive_batch.view(positive_batch.size(0), -1))

                    if selection_strategies[0] == 'HP':
                        # Hard positive
                        # positive_idx = torch.argmax(positive_dist)

                        _, positive_idx = torch.topk(positive_dist, 1)
                    else:
                        # Easy positive
                        # positive_idx = torch.argmin(positive_dist)
                        _, positive_idx = torch.topk(positive_dist, 1, largest=False)

                    positive_x = positive_batch[positive_idx]
                    x_m = torch.cat((anchor_x, positive_x), dim=0)
                    y_m = torch.tensor([m, m])
                else:
                    x_m = anchor_x
                    y_m = torch.tensor([m])

                if scenario in ['task', 'domain']:
                    self.add_instances_to_online_exemplar_sets(x_m, y_m, (y_m + len(uq) * (task - 1)).detach().cpu().numpy())
                else:
                    self.add_instances_to_online_exemplar_sets(x_m, y_m, y_m.detach().cpu().numpy())

                # Select negative instance
                negative_batch = x[mask_neg]
                negative_embed_batch = embeds[mask_neg]
                if negative_batch.size(0) != 0:
                    if use_embeddings:
                        anchor_batch = anchor_embed.expand(negative_embed_batch.size())
                        negative_dist = F.pairwise_distance(anchor_batch.view(anchor_batch.size(0), -1),
                                                            negative_embed_batch.view(negative_embed_batch.size(0), -1))
                    else:
                        anchor_batch = anchor_x.expand(negative_batch.size())
                        negative_dist = F.pairwise_distance(anchor_batch.view(anchor_batch.size(0), -1),
                                                                negative_batch.view(negative_batch.size(0), -1))

                    if selection_strategies[1] == 'HN':
                        # Hard negative
                        # if int(selection_strategies[2]) == 1:
                        #     negative_idx = torch.argmin(negative_dist)
                        #     negative_x = negative_batch[negative_idx].unsqueeze(dim=0)
                        #     negative_y = y[mask_neg][negative_idx].unsqueeze(dim=0)
                        # else:
                        _, negative_idx = torch.topk(negative_dist, int(selection_strategies[2]), largest=False)
                        negative_x = negative_batch[negative_idx]
                        negative_y = y[mask_neg][negative_idx]
                    elif selection_strategies[1] == 'SHN':
                        # Semi-hard negative
                        if use_embeddings:
                            positive_embed = positive_embed_batch[positive_idx].unsqueeze(dim=0)
                            dap = F.pairwise_distance(anchor_embed.view(anchor_x.size(0), -1),
                                                      positive_embed.view(positive_x.size(0), -1))
                        else:
                            dap = F.pairwise_distance(anchor_x.view(anchor_x.size(0), -1),
                                                          positive_x.view(positive_x.size(0), -1))
                        valid_shn_idx = negative_dist > dap
                        if valid_shn_idx.any():
                            shn_batch = negative_batch[valid_shn_idx]
                            negative_y_batch = y[mask_neg]
                            shn_y = negative_y_batch[valid_shn_idx]
                            # negative_idx = torch.argmin(negative_dist[valid_shn_idx])
                            _, negative_idx = torch.topk(negative_dist, int(selection_strategies[2]), largest=False)
                            negative_x = shn_batch[negative_idx]
                            negative_y = shn_y[negative_idx]
                        else:
                            # There is no semi-hard negative sample, ignore negative sample
                            negative_x = None
                            negative_y = None
                    else:
                        # Easy negative
                        # negative_idx = torch.argmax(negative_dist)
                        _, negative_idx = torch.topk(negative_dist, int(selection_strategies[2]))
                        negative_x = negative_batch[negative_idx]
                        negative_y = y[mask_neg][negative_idx]

                    if negative_x is not None and negative_y is not None:
                        if scenario in ['task', 'domain']:
                            self.add_instances_to_online_exemplar_sets(negative_x, negative_y,
                                                                       (negative_y + len(uq) * (task - 1)).detach().cpu().numpy())
                        else:
                            self.add_instances_to_online_exemplar_sets(negative_x, negative_y,
                                                                       negative_y.detach().cpu().numpy())

    def train_a_batch(self, x, y, scores=None, x_=None, y_=None, scores_=None, rnt=0.5,
                      active_classes=None, task=1, scenario='class',
                      loss_fn=None, replay_mode='none', triplet_selection='HP-HN',
                      use_embeddings=False,
                      teacher=None, otr_exemplars=False):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_/scores_]).

        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <tensor> batch of corresponding labels
        [scores]          None or <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x]
                            NOTE: only to be used for "BCE with distill" (only when scenario=="class")
        [x_]              None or (<list> of) <tensor> batch of replayed inputs
        [y_]              None or (<list> of) <tensor> batch of corresponding "replayed" labels
        [scores_]         None or (<list> of) <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x_]
        [rnt]             <number> in [0,1], relative importance of new task
        [active_classes]  None or (<list> of) <list> with "active" classes
        [task]            <int>, for setting task-specific mask'''

        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.optimizer.zero_grad()

        # Should gradient be computed separately for each task? (needed when a task-mask is combined with replay)
        gradient_per_task = True if ((self.mask_dict is not None) and (x_ is not None)) else False
        ##--(1)-- REPLAYED DATA --##

        if x_ is not None:
            # print(y_, task)
            # In the Task-IL scenario, [y_] or [scores_] is a list and [x_] needs to be evaluated on each of them
            # (in case of 'exact' or 'exemplar' replay, [x_] is also a list!
            TaskIL = (type(y_) == list) if (y_ is not None) else (type(scores_) == list)
            if not TaskIL:
                y_ = [y_]
                scores_ = [scores_]
                active_classes = [active_classes] if (active_classes is not None) else None
            n_replays = len(y_) if (y_ is not None) else len(scores_)

            # Prepare lists to store losses for each replay
            loss_replay = [None] * n_replays
            predL_r = [None] * n_replays
            distilL_r = [None] * n_replays

            # Run model (if [x_] is not a list with separate replay per task and there is no task-specific mask)
            if (not type(x_) == list) and (self.mask_dict is None):
                y_hat_all = self(x_)

            # Loop to evalute predictions on replay according to each previous task
            for replay_id in range(n_replays):

                # -if [x_] is a list with separate replay per task, evaluate model on this task's replay
                if (type(x_) == list) or (self.mask_dict is not None):
                    x_temp_ = x_[replay_id] if type(x_) == list else x_
                    if self.mask_dict is not None:
                        self.apply_XdGmask(task=replay_id + 1)
                    y_hat_all = self(x_temp_)

                # -if needed (e.g., Task-IL or Class-IL scenario), remove predictions for classes not in replayed task
                y_hat = y_hat_all if (active_classes is None) else y_hat_all[:, active_classes[replay_id]]

                # Calculate losses
                if (y_ is not None) and (y_[replay_id] is not None):
                    # if self.loss == 'otfl':
                    #     predL_r[replay_id], _ = loss_fn(x_[replay_id] if type(x_) == list else x_, y_hat, y_[replay_id])
                    # else:
                    if self.binaryCE:
                        binary_targets_ = utils.to_one_hot(y_[replay_id].cpu(), y_hat.size(1)).to(y_[replay_id].device)
                        predL_r[replay_id] = F.binary_cross_entropy_with_logits(
                            input=y_hat, target=binary_targets_, reduction='none'
                        ).sum(dim=1).mean()  # --> sum over classes, then average over batch
                    else:
                        predL_r[replay_id] = F.cross_entropy(y_hat, y_[replay_id], reduction='mean')
                if (scores_ is not None) and (scores_[replay_id] is not None):
                    # n_classes_to_consider = scores.size(1) #--> with this version, no zeroes are added to [scores]!
                    n_classes_to_consider = y_hat.size(1)  # --> zeros will be added to [scores] to make it this size!
                    kd_fn = utils.loss_fn_kd_binary if self.binaryCE else utils.loss_fn_kd
                    distilL_r[replay_id] = kd_fn(scores=y_hat[:, :n_classes_to_consider],
                                                 target_scores=scores_[replay_id], T=self.KD_temp)
                # Weigh losses
                if self.replay_targets == "hard":
                    loss_replay[replay_id] = predL_r[replay_id]
                elif self.replay_targets == "soft":
                    loss_replay[replay_id] = distilL_r[replay_id]

                # If needed, perform backward pass before next task-mask (gradients of all tasks will be accumulated)
                if gradient_per_task:
                    weight = 1 if self.AGEM else (1 - rnt)
                    weighted_replay_loss_this_task = weight * loss_replay[replay_id] / n_replays
                    weighted_replay_loss_this_task.backward()

            # Calculate total replay loss
            loss_replay = None if (x_ is None) else sum(loss_replay) / n_replays

        # If using A-GEM, calculate and store averaged gradient of replayed data
        if self.AGEM and x_ is not None:
            # Perform backward pass to calculate gradient of replayed batch (if not yet done)
            if not gradient_per_task:
                loss_replay.backward()
            # Reorganize the gradient of the replayed batch as a single vector
            grad_rep = []
            for p in self.parameters():
                if p.requires_grad:
                    grad_rep.append(p.grad.view(-1))
            grad_rep = torch.cat(grad_rep)
            # Reset gradients (with A-GEM, gradients of replayed batch should only be used as inequality constraint)
            self.optimizer.zero_grad()

        ##--(2)-- CURRENT DATA --##

        if x is not None:
            # If requested, apply correct task-specific mask
            if self.mask_dict is not None:
                self.apply_XdGmask(task=task)

            # Run model
            # y_hat = self(x)

            if self.experiment not in ['splitMNIST', 'permMNIST', 'rotMNIST']:
                embeds = self.fcE(x)
            else:
                embeds = self.fcE(self.flatten(x))
            y_hat = self.classifier(embeds)

            if teacher is not None:
                if teacher.is_ready_distill:
                    # with torch.no_grad():
                    #     teacher.eval()
                    y_hat_teacher = teacher(x)

                else:
                    y_hat_teacher = None
            else:
                y_hat_teacher = None

            # -if needed, remove predictions for classes not in current task
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                y_hat = y_hat[:, class_entries]
                if y_hat_teacher is not None:
                    y_hat_teacher = y_hat_teacher[:, class_entries]

            # Compute distillation loss from teacher outputs
            if y_hat_teacher is not None:
                # teacherL = nn.KLDivLoss()(F.log_softmax(y_hat / self.KD_temp, dim=1),
                #                           F.softmax(y_hat_teacher / self.KD_temp, dim=1)) \
                #            * (self.alpha_t * self.KD_temp * self.KD_temp) \
                #            + F.cross_entropy(y_hat, y) * (1. - self.alpha_t)
                with torch.no_grad():
                    y_hat_ensemble = 0.5 * (y_hat + y_hat_teacher)

                teacherL = nn.KLDivLoss()(F.log_softmax(y_hat_teacher / self.KD_temp, dim=1),
                                          F.softmax(y_hat_ensemble / self.KD_temp, dim=1)) \
                           * (self.KD_temp * self.KD_temp)

                studentL = nn.KLDivLoss()(F.log_softmax(y_hat / self.KD_temp, dim=1),
                                          F.softmax(y_hat_ensemble / self.KD_temp, dim=1)) \
                           * (self.KD_temp * self.KD_temp)

            else:
                teacherL = None
                studentL = None

            # Calculate prediction loss
            if self.loss in ['ofl', 'otfl', 'fgfl', 'gbfg']:
                predL, selected_data = loss_fn(x, y_hat, y)
                if replay_mode == 'online':
                    for k in selected_data.keys():
                        if scenario in ['task', 'domain']:  # Task-IL scenario
                            self.add_instances_to_online_exemplar_sets(selected_data[k][0], selected_data[k][1],
                                                                       k + len(selected_data.keys()) * (task - 1))
                        else:
                            self.add_instances_to_online_exemplar_sets(selected_data[k][0], selected_data[k][1], k)

            elif self.loss in ['focal', 'ce']:
                predL = loss_fn(y_hat, y)
            else:
                if self.binaryCE:
                    # -binary prediction loss
                    binary_targets = utils.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
                    if self.binaryCE_distill and (scores is not None):
                        classes_per_task = int(y_hat.size(1) / task)
                        binary_targets = binary_targets[:, -(classes_per_task):]
                        binary_targets = torch.cat([torch.sigmoid(scores / self.KD_temp), binary_targets], dim=1)
                    y_score = F.binary_cross_entropy_with_logits(
                        input=y_hat, target=binary_targets, reduction='none'
                    ).sum(dim=1) # --> sum over classes,
                    predL = None if y is None else y_score.mean()  # average over batch

                    self.select_triplets(embeds, y_score, x, y, triplet_selection, task, scenario, use_embeddings)
                    if otr_exemplars:
                        # softmax_score = F.cross_entropy(input=y_hat, target=y, reduction='none')
                        # self.select_triplets(embeds, softmax_score, x, y, triplet_selection, task, scenario, use_embeddings)
                        # Update class mean
                        self.compute_class_means(x, y)
                else:
                    # -multiclass prediction loss
                    y_score = F.cross_entropy(input=y_hat, target=y, reduction='none')
                    predL = None if y is None else y_score.mean()

                    if replay_mode == 'online':
                        self.select_triplets(embeds, y_score, x, y, triplet_selection, task, scenario, use_embeddings)

            # Weigh losses
            if teacherL is not None and studentL is not None:
                loss_cur = rnt * predL + (1 - rnt) * (1/2) * (teacherL + studentL)
            else:
                loss_cur = predL

            # Calculate training-precision
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)

            # If backward passes are performed per task (e.g., XdG combined with replay), perform backward pass
            if gradient_per_task:
                weighted_current_loss = rnt * loss_cur
                weighted_current_loss.backward()
        else:
            precision = predL = None
            # -> it's possible there is only "replay" [e.g., for offline with task-incremental learning]

        # Combine loss from current and replayed batch
        if x_ is None or self.AGEM:
            loss_total = loss_cur
        else:
            loss_total = loss_replay if (x is None) else rnt * loss_cur + (1 - rnt) * loss_replay

        ##--(3)-- ALLOCATION LOSSES --##

        # Add SI-loss (Zenke et al., 2017)
        surrogate_loss = self.surrogate_loss()
        if self.si_c > 0:
            loss_total += self.si_c * surrogate_loss

        # Add EWC-loss
        ewc_loss = self.ewc_loss()
        if self.ewc_lambda > 0:
            loss_total += self.ewc_lambda * ewc_loss

        # Backpropagate errors (if not yet done)
        if not gradient_per_task:
            loss_total.backward()

        # If using A-GEM, potentially change gradient:
        if self.AGEM and x_ is not None:
            # -reorganize gradient (of current batch) as single vector
            grad_cur = []
            for p in self.parameters():
                if p.requires_grad:
                    grad_cur.append(p.grad.view(-1))
            grad_cur = torch.cat(grad_cur)
            # -check inequality constrain
            angle = (grad_cur * grad_rep).sum()
            if angle < 0:
                # -if violated, project the gradient of the current batch onto the gradient of the replayed batch ...
                length_rep = (grad_rep * grad_rep).sum()
                grad_proj = grad_cur - (angle / length_rep) * grad_rep
                # -...and replace all the gradients within the model with this projected gradient
                index = 0
                for p in self.parameters():
                    if p.requires_grad:
                        n_param = p.numel()  # number of parameters in [p]
                        p.grad.copy_(grad_proj[index:index + n_param].view_as(p))
                        index += n_param

        # Take optimization-step
        self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'loss_current': loss_cur.item() if x is not None else 0,
            'loss_replay': loss_replay.item() if (x_ is not None and loss_replay is not None) else 0,
            'pred': predL.item() if predL is not None else 0,
            'pred_r': sum(predL_r).item() / n_replays if (x_ is not None and predL_r[0] is not None) else 0,
            'distil_r': sum(distilL_r).item() / n_replays if (x_ is not None and distilL_r[0] is not None) else 0,
            'ewc': ewc_loss.item(), 'si_loss': surrogate_loss.item(),
            'precision': precision if precision is not None else 0.,
        }

    def train_epoch(self, train_loader, criterion, optimizer, epoch):
        self.train()
        for batch_idx, batch in enumerate(train_loader):
            x, y = batch
            x, y = x.to(self._device()), y.to(self._device())

            optimizer.zero_grad()
            y_hat = self(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

    def valid_epoch(self, val_loader, criterion):
        valid_losses = []
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader, 0):
                x, y = batch
                x, y = x.to(self._device()), y.to(self._device())
                y_hat = self(x)
                valid_loss = criterion(y_hat, y)
                valid_losses.append(valid_loss.item())
        self.train()
        return valid_losses

