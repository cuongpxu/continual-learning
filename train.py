import torch
import tqdm
import copy
import utils
import os
import numpy as np
import threading
import uuid
from data import SubDataset, ExemplarDataset, OnlineExemplarDataset, TransformedDataset
from continual_learner import ContinualLearner
from es import EarlyStopping
from torch import optim
from torch.utils.data import ConcatDataset, random_split
from optim import optimizer
from torch.utils.tensorboard import SummaryWriter


def train_cl(model, teacher, train_datasets, replay_mode="none", scenario="class", classes_per_task=None, iters=2000,
             batch_size=32, generator=None, gen_iters=0,
             gen_loss_cbs=list(), loss_cbs=list(), eval_cbs=list(), sample_cbs=list(),
             use_exemplars=True, add_exemplars=False, metric_cbs=list(),
             params_dict=None):
    '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

    [model]             <nn.Module> main model to optimize across all tasks
    [train_datasets]    <list> with for each task the training <DataSet>
    [replay_mode]       <str>, choice from "generative", "exact", "current", "offline" and "none"
    [scenario]          <str>, choice from "task", "domain" and "class"
    [classes_per_task]  <int>, # of classes per task
    [iters]             <int>, # of optimization-steps (i.e., # of batches) per task
    [generator]         None or <nn.Module>, if a seperate generative model should be trained (for [gen_iters] per task)
    [*_cbs]             <list> of call-back functions to evaluate training-progress'''

    # Set model in training-mode
    params_dict['use_exemplars'] = use_exemplars
    model.train()
    torch.autograd.set_detect_anomaly(True)
    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Initiate possible sources for replay (no replay for 1st task)
    Exact = Generative = Current = False
    previous_model = None

    # Register starting param-values (needed for "intelligent synapses").
    if isinstance(model, ContinualLearner) and (model.si_c > 0):
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())

    # Loop over all tasks.
    for task, train_dataset in enumerate(train_datasets, 1):
        # Re-train teacher in every task
        if teacher is not None and not params_dict['online_kd']:
            teacher.is_offline_training = False
            if task > 1:
                teacher.is_ready_distill = True

        # If offline replay-setting, create large database of all tasks so far
        if replay_mode == "offline" and (not scenario == "task"):
            train_dataset = ConcatDataset(train_datasets[:task])
        # -but if "offline"+"task"-scenario: all tasks so far included in 'exact replay' & no current batch
        if replay_mode == "offline" and scenario == "task":
            Exact = True
            previous_datasets = train_datasets

        # Add exemplars (if available) to current dataset (if requested)
        if add_exemplars and task > 1:
            target_transform = (lambda y, x=classes_per_task: y % x) if scenario == "domain" else None
            if params_dict['use_otr'] or not params_dict['mem_online']:
                exemplar_sets = []
                for c in sorted(model.online_exemplar_sets):
                    exemplar_sets.append(model.online_exemplar_sets[c][0])

                exemplar_dataset = ExemplarDataset(exemplar_sets, target_transform=target_transform)
                training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
            else:
                exemplar_dataset = ExemplarDataset(model.exemplar_sets, target_transform=target_transform)
                training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
        else:
            training_dataset = train_dataset

        # Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
        if isinstance(model, ContinualLearner) and (model.si_c > 0):
            W = {}
            p_old = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()

        # Find [active_classes]
        active_classes = None  # -> for Domain-IL scenario, always all classes are active
        if scenario == "task":
            # -for Task-IL scenario, create <list> with for all tasks so far a <list> with the active classes
            active_classes = [list(range(classes_per_task * i, classes_per_task * (i + 1))) for i in range(task)]
        elif scenario == "class":
            # -for Class-IL scenario, create one <list> with active classes of all tasks so far
            active_classes = list(range(classes_per_task * task))

        # Reset state of optimizer(s) for every task (if requested)
        if model.optim_type == "adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
        if (generator is not None) and generator.optim_type == "adam_reset":
            generator.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

        # Initialize # iters left on current data-loader(s)
        iters_left = iters_left_previous = 1
        if scenario == "task":
            up_to_task = task if replay_mode == "offline" else task - 1
            iters_left_previous = [1] * up_to_task
            data_loader_previous = [None] * up_to_task

        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters + 1))
        if generator is not None:
            progress_gen = tqdm.tqdm(range(1, gen_iters + 1))

        # Loop over all iterations
        iters_to_use = iters if (generator is None) else max(iters, gen_iters)
        for batch_index in range(1, iters_to_use + 1):

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left == 0:
                data_loader = iter(utils.get_data_loader(training_dataset, batch_size, cuda=cuda, drop_last=True))
                # NOTE:  [train_dataset]  is training-set of current task
                #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                iters_left = len(data_loader)
            if Exact:
                if scenario == "task":
                    up_to_task = task if replay_mode == "offline" else task - 1
                    batch_size_replay = int(np.ceil(batch_size / up_to_task)) if (up_to_task > 1) else batch_size
                    # -in Task-IL scenario, need separate replay for each task
                    for task_id in range(up_to_task):
                        batch_size_to_use = min(batch_size_replay, len(previous_datasets[task_id]))
                        iters_left_previous[task_id] -= 1
                        if iters_left_previous[task_id] == 0:
                            data_loader_previous[task_id] = iter(utils.get_data_loader(
                                train_datasets[task_id], batch_size_to_use, cuda=cuda, drop_last=True
                            ))
                            iters_left_previous[task_id] = len(data_loader_previous[task_id])
                else:
                    iters_left_previous -= 1
                    if iters_left_previous == 0:
                        batch_size_to_use = min(batch_size, len(ConcatDataset(previous_datasets)))
                        data_loader_previous = iter(utils.get_data_loader(ConcatDataset(previous_datasets),
                                                                          batch_size_to_use, cuda=cuda, drop_last=True))
                        iters_left_previous = len(data_loader_previous)

            # -----------------Collect data------------------#

            #####-----CURRENT BATCH-----#####
            if replay_mode == "offline" and scenario == "task":
                x = y = scores = None
            else:
                x, y = next(data_loader)  # --> sample training data of current task
                y = y - classes_per_task * (task - 1) if scenario == "task" else y  # --> ITL: adjust y-targets to 'active range'
                y = y.long()
                x, y = x.to(device), y.to(device)  # --> transfer them to correct device
                # If --bce, --bce-distill & scenario=="class", calculate scores of current batch with previous model
                binary_distillation = hasattr(model, "binaryCE") and model.binaryCE and model.binaryCE_distill
                if binary_distillation and scenario == "class" and (previous_model is not None):
                    with torch.no_grad():
                        scores = previous_model(x)[:, :(classes_per_task * (task - 1))]
                else:
                    scores = None

            #####-----REPLAYED BATCH-----#####
            if not Exact and not Generative and not Current:
                x_ = y_ = scores_ = None  # -> if no replay

            ##-->> Exact Replay <<--##
            if Exact:
                scores_ = None
                if scenario in ("domain", "class"):
                    # Sample replayed training data, move to correct device
                    x_, y_ = next(data_loader_previous)
                    x_ = x_.to(device)
                    y_ = y_.to(device) if (model.replay_targets == "hard") else None
                    # If required, get target scores (i.e, [scores_]         -- using previous model, with no_grad()
                    if model.replay_targets == "soft":
                        with torch.no_grad():
                            scores_ = previous_model(x_)
                        scores_ = scores_[:, :(classes_per_task * (task - 1))] if scenario == "class" else scores_
                        # -> when scenario=="class", zero probabilities will be added in the [utils.loss_fn_kd]-function
                elif scenario == "task":
                    # Sample replayed training data, wrap in (cuda-)Variables and store in lists
                    x_ = list()
                    y_ = list()
                    up_to_task = task if replay_mode == "offline" else task - 1
                    for task_id in range(up_to_task):
                        x_temp, y_temp = next(data_loader_previous[task_id])
                        x_.append(x_temp.to(device))
                        # -only keep [y_] if required (as otherwise unnecessary computations will be done)
                        if model.replay_targets == "hard":
                            y_temp = y_temp - (classes_per_task * task_id)  # -> adjust y-targets to 'active range'
                            y_.append(y_temp.to(device))
                        else:
                            y_.append(None)
                    # If required, get target scores (i.e, [scores_]         -- using previous model
                    if (model.replay_targets == "soft") and (previous_model is not None):
                        scores_ = list()
                        for task_id in range(up_to_task):
                            with torch.no_grad():
                                scores_temp = previous_model(x_[task_id])
                            scores_temp = scores_temp[:,
                                          (classes_per_task * task_id):(classes_per_task * (task_id + 1))]
                            scores_.append(scores_temp)

            ##-->> Generative / Current Replay <<--##
            if Generative or Current:
                # Get replayed data (i.e., [x_]) -- either current data or use previous generator
                x_ = x if Current else previous_generator.sample(batch_size)

                # Get target scores and labels (i.e., [scores_] / [y_]) -- using previous model, with no_grad()
                # -if there are no task-specific mask, obtain all predicted scores at once
                if (not hasattr(previous_model, "mask_dict")) or (previous_model.mask_dict is None):
                    with torch.no_grad():
                        all_scores_ = previous_model(x_)
                # -depending on chosen scenario, collect relevant predicted scores (per task, if required)
                if scenario in ("domain", "class") and (
                        (not hasattr(previous_model, "mask_dict")) or (previous_model.mask_dict is None)
                ):
                    scores_ = all_scores_[:, :(classes_per_task * (task - 1))] if scenario == "class" else all_scores_
                    _, y_ = torch.max(scores_, dim=1)
                else:
                    # NOTE: it's possible to have scenario=domain with task-mask (so actually it's the Task-IL scenario)
                    # -[x_] needs to be evaluated according to each previous task, so make list with entry per task
                    scores_ = list()
                    y_ = list()
                    for task_id in range(task - 1):
                        # -if there is a task-mask (i.e., XdG is used), obtain predicted scores for each task separately
                        if hasattr(previous_model, "mask_dict") and previous_model.mask_dict is not None:
                            previous_model.apply_XdGmask(task=task_id + 1)
                            with torch.no_grad():
                                all_scores_ = previous_model(x_)
                        if scenario == "domain":
                            temp_scores_ = all_scores_
                        else:
                            temp_scores_ = all_scores_[:,
                                           (classes_per_task * task_id):(classes_per_task * (task_id + 1))]
                        _, temp_y_ = torch.max(temp_scores_, dim=1)
                        scores_.append(temp_scores_)
                        y_.append(temp_y_)

                # Only keep predicted y/scores if required (as otherwise unnecessary computations will be done)
                y_ = y_ if (model.replay_targets == "hard") else None
                scores_ = scores_ if (model.replay_targets == "soft") else None

            # ---> Train MAIN MODEL
            if batch_index <= iters:
                # print('Training batch: {}'.format(batch_index))
                # Train the main model with this batch
                for e in range(params_dict['epochs']):
                    loss_dict = model.train_a_batch(x, y, x_=x_, y_=y_, scores=scores, scores_=scores_,
                                                    active_classes=active_classes, task=task, rnt=1. / task,
                                                    scenario=scenario, teacher=teacher,
                                                    params_dict=params_dict, epoch=e)
                    if params_dict['online_kd']:
                        # Incremental training teacher model
                        teacher_loss_dict = teacher.train_a_batch(x, y, x_=x_, y_=y_, scores=scores, scores_=scores_,
                                                    active_classes=active_classes, task=task, rnt=1. / task,
                                                    scenario=scenario, teacher=model,
                                                    params_dict=params_dict)

                    if params_dict['update_teacher_kd']:
                        if teacher is not None and teacher.is_ready_distill and task > 1:
                            teacher.train_via_KD(model, x,  params_dict['distill_type'], active_classes)

                # Update running parameter importance estimates in W
                if isinstance(model, ContinualLearner) and (model.si_c > 0):
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            if p.grad is not None:
                                W[n].add_(-p.grad * (p.detach() - p_old[n]))
                            p_old[n] = p.detach().clone()

                # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, task=task)
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(model, batch_index, task=task)
                if model.label == "VAE":
                    for sample_cb in sample_cbs:
                        if sample_cb is not None:
                            sample_cb(model, batch_index, task=task)

            # ---> Train GENERATOR
            if generator is not None and batch_index <= gen_iters:

                # Train the generator with this batch
                loss_dict = generator.train_a_batch(x, y, x_=x_, y_=y_, scores_=scores_, active_classes=active_classes,
                                                    task=task, rnt=1. / task)

                # Fire callbacks on each iteration
                for loss_cb in gen_loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress_gen, batch_index, loss_dict, task=task)
                for sample_cb in sample_cbs:
                    if sample_cb is not None:
                        sample_cb(generator, batch_index, task=task)

            # ---> Train Teacher in offline mode
            if teacher is not None and (replay_mode == 'online' and model.check_full_memory())\
                    and not params_dict['online_kd']:
                if not teacher.is_offline_training:
                    # Start training a teacher in offline mode from memory
                    teacher.is_offline_training = True
                    teacher.is_ready_distill = False

                    # Get dataset from online memory
                    if scenario in ['task', 'domain']:
                        memory_datasets = []
                        for task_id in range(task):
                            classes = np.arange((classes_per_task * task_id), (classes_per_task * (task_id + 1)))
                            for c in classes:
                                memory_datasets.append(
                                    OnlineExemplarDataset(model.online_exemplar_sets[c])
                                )
                    else:
                        memory_datasets = []
                        for c in sorted(model.online_exemplar_sets):
                            memory_datasets.append(OnlineExemplarDataset(model.online_exemplar_sets[c]))

                    teacher_dataset = ConcatDataset(memory_datasets)
                    # params_dict['teacher_lr'] = model.optimizer.param_groups[0]['lr']
                    params_dict['batch_size'] = batch_size
                    params_dict['cuda'] = cuda
                    params_dict['task'] = task

                    teacherThread = TeacherThread(1, teacher_dataset, teacher, active_classes, params_dict)
                    teacherThread.start()

                    teacherThread.join()

        ##----------> UPON FINISHING EACH TASK...

        # Close progres-bar(s)
        progress.close()
        if generator is not None:
            progress_gen.close()

        # EWC: estimate Fisher Information matrix (FIM) and update term for quadratic penalty
        if isinstance(model, ContinualLearner) and (model.ewc_lambda > 0):
            # -find allowed classes
            allowed_classes = list(
                range(classes_per_task * (task - 1), classes_per_task * task)
            ) if scenario == "task" else (list(range(classes_per_task * task)) if scenario == "class" else None)
            # -if needed, apply correct task-specific mask
            if model.mask_dict is not None:
                model.apply_XdGmask(task=task)
            # -estimate FI-matrix
            model.estimate_fisher(training_dataset, allowed_classes=allowed_classes)

        # SI: calculate and update the normalized path integral
        if isinstance(model, ContinualLearner) and (model.si_c > 0):
            model.update_omega(W, model.epsilon)

        # EXEMPLARS: update exemplar sets
        if (add_exemplars or use_exemplars) or replay_mode == "exemplars":
            if not params_dict['use_otr'] and not params_dict['mem_online']:
                exemplars_per_class = int(np.floor(model.memory_budget / (classes_per_task * task)))
                # reduce examplar-sets
                model.reduce_exemplar_sets(exemplars_per_class)
                # for each new class trained on, construct examplar-set
                new_classes = list(range(classes_per_task)) if scenario == "domain" else list(
                    range(classes_per_task * (task - 1),
                          classes_per_task * task))
                for class_id in new_classes:
                    # create new dataset containing only all examples of this class
                    class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                    # based on this dataset, construct new exemplar-set for this class
                    model.construct_exemplar_set(dataset=class_dataset, n=exemplars_per_class)
                model.compute_means = True
            else:
                model.compute_means = True
        # Calculate statistics required for metrics
        for metric_cb in metric_cbs:
            if metric_cb is not None:
                metric_cb(model, iters, task=task, otr_exemplars=params_dict['otr_exemplars'])

        # REPLAY: update source for replay
        if previous_model is not None:
            previous_model.load_state_dict(model.state_dict())
            previous_model.eval()
        else:
            previous_model = copy.copy(model)
            previous_model.load_state_dict(model.state_dict())
            previous_model.eval()

        if replay_mode == 'generative':
            Generative = True
            previous_generator = copy.deepcopy(generator).eval() if generator is not None else previous_model
        elif replay_mode == 'current':
            Current = True
        elif replay_mode in ('exemplars', 'exact', 'online'):
            Exact = True
            if replay_mode == "exact":
                previous_datasets = train_datasets[:task]
            elif replay_mode == 'online':
                if not add_exemplars:
                    if scenario in ['task', 'domain']:
                        previous_datasets = []
                        for task_id in range(task):
                            classes = np.arange((classes_per_task * task_id), (classes_per_task * (task_id + 1)))
                            for c in classes:
                                previous_datasets.append(
                                    OnlineExemplarDataset(model.online_exemplar_sets[c])
                                )
                    else:
                        previous_datasets = []
                        for c in sorted(model.online_exemplar_sets):
                            previous_datasets.append(OnlineExemplarDataset(model.online_exemplar_sets[c]))
                else:
                    Exact = False
            else:
                if scenario == "task":
                    previous_datasets = []
                    for task_id in range(task):
                        previous_datasets.append(
                            ExemplarDataset(
                                model.exemplar_sets[
                                (classes_per_task * task_id):(classes_per_task * (task_id + 1))],
                                target_transform=lambda y, x=classes_per_task * task_id: y + x)
                        )
                else:
                    target_transform = (lambda y, x=classes_per_task: y % x) if scenario == "domain" else None
                    previous_datasets = [
                        ExemplarDataset(model.exemplar_sets, target_transform=target_transform)]


def training_teacher(teacher_dataset, teacher, active_classes, params_dict):
    # Split dataset into train and val sets
    torch.autograd.set_detect_anomaly(True)
    teacher_split = params_dict['teacher_split']
    mem_train_size = int(teacher_split * len(teacher_dataset))
    mem_train_set, mem_val_set = random_split(teacher_dataset, [mem_train_size, len(teacher_dataset) - mem_train_size],
                                              torch.Generator().manual_seed(params_dict['seed']))
    if params_dict['teacher_augment'] is not None:
        mem_train_set = TransformedDataset(mem_train_set, transform=params_dict['teacher_augment'])
    mem_train_loader = utils.get_data_loader(mem_train_set, batch_size=params_dict['batch_size'],
                                             shuffle=True, drop_last=False, cuda=params_dict['cuda'])
    mem_val_loader = utils.get_data_loader(mem_val_set, batch_size=params_dict['batch_size'],
                                           shuffle=False, drop_last=False, cuda=params_dict['cuda'])
    if params_dict['teacher_opt'] == 'Adam':
        teacher_optimizer = optim.Adam(teacher.parameters(), lr=params_dict['teacher_lr'], betas=(0.9, 0.999))
    elif params_dict['teacher_opt'] == 'AdLR':
        teacher_optimizer = optimizer.AdLR(teacher.parameters(), lr=params_dict['teacher_lr'])
    elif params_dict['teacher_opt'] == 'eAdLR':
        teacher_optimizer = optimizer.eAdLR(teacher.parameters(), lr=params_dict['teacher_lr'])
    else:
        teacher_optimizer = optim.SGD(teacher.parameters(), lr=params_dict['teacher_lr'], momentum=0.9)

    if params_dict['use_scheduler']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(teacher_optimizer, 'min', patience=3)

    if params_dict['teacher_loss'] == 'CE':
        teacher_criterion = torch.nn.CrossEntropyLoss()
    else:
        teacher_criterion = torch.nn.BCEWithLogitsLoss()
    id = uuid.uuid1()
    early_stopping = EarlyStopping(model_name=id.hex, verbose=False)
    # utils.make_dirs('%s/board' % params_dict['r_dir'])
    # writer = SummaryWriter('%s/board/%s_%s' % (params_dict['r_dir'], params_dict['stamp'], params_dict['task']))

    # Training teacher
    tk = tqdm.tqdm(range(1, params_dict['teacher_epochs']), leave=False)
    tk.set_description('<Teacher> ')
    for epoch in tk:
        params_dict['epoch'] = epoch
        tlosses = teacher.train_epoch(mem_train_loader, teacher_criterion, teacher_optimizer,
                                      active_classes, params_dict)
        vlosses = teacher.valid_epoch(mem_val_loader, teacher_criterion,
                                      active_classes, params_dict)
        tk.set_description('<Teacher> | training_loss : {:.5f} | validation_loss: {:.5f}'
                           .format(np.average(tlosses), np.average(vlosses)), refresh=True)
        if params_dict['use_scheduler']:
            scheduler.step(np.average(vlosses))
        early_stopping(np.average(vlosses), teacher, epoch)
        if early_stopping.early_stop:
            # print("Teacher early stopping detected")
            # Notify solver to distill from teacher
            # teacher.is_offline_training = False
            # teacher.is_ready_distill = True
            # Using best model from last check point
            teacher.load_state_dict(torch.load(early_stopping.model_name))
            break

    # teacher.is_offline_training = False
    teacher.is_ready_distill = False
    if os.path.exists(early_stopping.model_name):
        os.remove(early_stopping.model_name)


class TeacherThread(threading.Thread):
    def __init__(self, threadID, teacher_dataset, teacher, active_classes, params_dict):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.teacher_dataset = teacher_dataset
        self.teacher = teacher
        self.active_classes = active_classes
        self.params_dict = params_dict

    def run(self):
        training_teacher(self.teacher_dataset, self.teacher, self.active_classes, self.params_dict)
