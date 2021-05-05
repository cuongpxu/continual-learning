import argparse
import os
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from param_stamp import get_param_stamp_from_args
from param_values import set_default_values

description = 'Compare CL strategies using various metrics on each scenario of permuted or split MNIST.'
parser = argparse.ArgumentParser('./create_result.py', description=description)
parser.add_argument('--form', type=str, default='NIPS', help='Table form')

parser.add_argument('--seed', type=int, default=1, help='[first] random seed (for each random-module used)')
parser.add_argument('--n-seeds', type=int, default=5, help='how often to repeat?')
parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results-dir', type=str, default='../benchmark_epochs', dest='r_dir', help="default: %(default)s")

# expirimental task parameters.
task_params = parser.add_argument_group('Task Parameters')
task_params.add_argument('--experiment', type=str, default='splitMNIST',
                         choices=['rotMNIST', 'permMNIST', 'splitMNIST', 'CIFAR10', 'CIFAR100'])
task_params.add_argument('--scenario', type=str, default='task', choices=['task', 'domain', 'class'])
task_params.add_argument('--tasks', type=int, help='number of tasks')
# specify loss functions to be used
loss_params = parser.add_argument_group('Loss Parameters')
loss_params.add_argument('--loss', type=str, default='none',
                         choices=['otfl', 'fgfl', 'focal', 'ce', 'gbfg', 'none'])
loss_params.add_argument('--bce', action='store_true', help="use binary (instead of multi-class) classication loss")

# model architecture parameters
model_params = parser.add_argument_group('Parameters Main Model')
model_params.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
model_params.add_argument('--fc-units', type=int, metavar="N", help="# of units in first fc-layers")
model_params.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
model_params.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
model_params.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu"])
model_params.add_argument('--singlehead', action='store_true', help="for Task-IL: use a 'single-headed' output layer   "
                                                                    " (instead of a 'multi-headed' one)")
model_params.add_argument('--use_teacher', action='store_true', help='Using an offline teacher for distill from memory')
model_params.add_argument('--teacher_epochs', type=int, default=100, help='number of epochs to train teacher')
model_params.add_argument('--teacher_loss', type=str, default='CE', help='teacher loss function')
model_params.add_argument('--teacher_split', type=float, default=0.8, help='split ratio for teacher training')
model_params.add_argument('--teacher_opt', type=str, default='Adam', help='teacher optimizer')
model_params.add_argument('--use_scheduler', action='store_true', help='Using learning rate scheduler for teacher')
model_params.add_argument('--use_augment', action='store_true', help='Using data augmentation for training teacher')
model_params.add_argument('--distill_type', type=str, default='T', choices=['T', 'TS', 'E', 'ET', 'ETS'])
model_params.add_argument('--multi_negative', type=utils.str_to_bool, default=False)
model_params.add_argument('--update_teacher_kd', type=utils.str_to_bool, default=True)
model_params.add_argument('--online_kd', type=utils.str_to_bool, default=False)
model_params.add_argument('--epochs', type=int, default=1)
# training hyperparameters / initialization
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--iters', type=int, help="# batches to optimize solver")
train_params.add_argument('--lr', type=float, help="learning rate")
train_params.add_argument('--batch', type=int, default=128, help="batch-size")
train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')

# "memory replay" parameters
replay_params = parser.add_argument_group('Replay Parameters')
replay_choices = ['offline', 'exact', 'generative', 'none', 'current', 'exemplars', 'online']
replay_params.add_argument('--replay', type=str, default='none', choices=replay_choices)
replay_params.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
replay_params.add_argument('--otr_exemplars', action='store_true', help="use otr exemplars instead of random")
replay_params.add_argument('--triplet-selection', type=str, default='HP-HN-1', help="Triplet selection strategy")
replay_params.add_argument('--use-embeddings', type=bool, default=False,
                          help="use embeddings space for otr exemplars instead of features space")
replay_params.add_argument('--mem_online', type=utils.str_to_bool, default=False, help='icarl using online exemplar mamagement')
# -generative model parameters (if separate model)
genmodel_params = parser.add_argument_group('Generative Model Parameters')
genmodel_params.add_argument('--g-z-dim', type=int, default=100, help='size of latent representation (default: 100)')
genmodel_params.add_argument('--g-fc-lay', type=int, help='[fc_layers] in generator (default: same as classifier)')
genmodel_params.add_argument('--g-fc-uni', type=int, help='[fc_units] in generator (default: same as classifier)')
# - hyper-parameters for generative model (if separate model)
gen_params = parser.add_argument_group('Generator Hyper Parameters')
gen_params.add_argument('--g-iters', type=int, help="# batches to train generator (default: as classifier)")
gen_params.add_argument('--lr-gen', type=float, help="learning rate generator (default: lr)")

# "memory allocation" parameters
cl_params = parser.add_argument_group('Memory Allocation Parameters')
cl_params.add_argument('--lambda', type=float, dest="ewc_lambda", help="--> EWC: regularisation strength")
cl_params.add_argument('--o-lambda', type=float, help="--> online EWC: regularisation strength")
cl_params.add_argument('--fisher-n', type=int, help="--> EWC: sample size estimating Fisher Information")
cl_params.add_argument('--gamma', type=float, help="--> EWC: forgetting coefficient (for 'online EWC')")
cl_params.add_argument('--emp-fi', action='store_true', help="--> EWC: estimate FI with provided labels")
cl_params.add_argument('--c', type=float, dest="si_c", help="--> SI: regularisation strength")
cl_params.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="--> SI: dampening parameter")
cl_params.add_argument('--gating-prop', type=float, metavar="PROP", help="--> XdG: prop neurons per layer to gate")

# iCaRL parameters
icarl_params = parser.add_argument_group('iCaRL Parameters')
icarl_params.add_argument('--budget', type=int, default=2000, dest="budget", help="how many exemplars can be stored?")
icarl_params.add_argument('--herding', action='store_true', help="use herding to select exemplars (instead of random)")
icarl_params.add_argument('--use-exemplars', action='store_true', help="use stored exemplars for classification?")
icarl_params.add_argument('--norm-exemplars', action='store_true', help="normalize features/averages of exemplars")

# evaluation parameters
eval_params = parser.add_argument_group('Evaluation Parameters')
eval_params.add_argument('--time', action='store_true', help="keep track of total training time")
eval_params.add_argument('--pdf', action='store_true', help="generate pdfs for individual experiments")
eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
eval_params.add_argument('--prec-n', type=int, default=1024, help="# samples for evaluating solver's precision")
eval_params.add_argument('--sample-n', type=int, default=64, help="# images to show")


def set_params(args, a):
    if a == 'None':
        args.replay = "none"
    elif a == 'Offline':
        args.replay = "offline"
    elif a == 'EWC':
        args.replay = 'none'
        args.ewc = True
    elif a == 'o-EWC':
        args.replay = 'none'
        args.ewc = True
        args.online = True
        args.ewc_lambda = args.o_lambda
    elif a == 'SI':
        args.replay = 'none'
        args.si = True
    elif a == 'LwF':
        args.replay = "current"
        args.distill = True
    elif a == 'GR':
        args.replay = "generative"
        args.distill = False
        if args.experiment in ['CIFAR10', 'CIFAR100']:
            args.lr_gen = 0.0003
    elif a == 'GR+distill':
        args.replay = "generative"
        args.distill = True
        if args.experiment in ['CIFAR10', 'CIFAR100']:
            args.lr_gen = 0.0003
    elif a == 'A-GEM':
        args.replay = "exemplars"
        args.distill = False
        args.agem = True
        args.otr_exemplars = False
    elif a == 'ER':
        args.replay = "exemplars"
        args.budget = 2000
        args.otr_exemplars = False
        args.mem_online = True
    elif a == 'OTR':
        args.replay = 'online'
        args.budget = 2000
        args.triplet_selection = 'HP-HN-1'
        args.bce = True
        if args.scenario == 'class':
            args.bce_distill = True
        args.use_embeddings = False
        args.multi_negative = False
        args.add_exemplars = False
    elif a == 'iCaRL':
        args.bce = True
        args.bce_distill = True
        args.use_exemplars = True
        args.add_exemplars = False
        args.herding = True
        args.norm_exemplars = True
        args.mem_online = True
    else:
        args.replay = 'online'
        args.budget = 2000
        args.use_teacher = True
        args.use_embeddings = False
        args.triplet_selection = 'HP-HN-1'
        args.teacher_epochs = 100
        args.teacher_loss = 'CE'
        args.teacher_split = 0.8
        args.teacher_opt = 'Adam'
        args.use_scheduler = False
        args.distill_type = 'E'
        args.multi_negative = False
        args.use_augment = False


def get_results(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run; if not do so
    if os.path.isfile("{}/dict-{}.pkl".format(args.r_dir, param_stamp)):
        print("{}: already run".format(param_stamp))
    else:
        print("{}: ...missing...".format(param_stamp))
        raise Exception('Missing results !!!!')
    # -get results-dict
    dict = utils.load_object("{}/dict-{}".format(args.r_dir, param_stamp))
    # -get test accuracy
    fileName = '{}/prec-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    prec = float(file.readline())
    file.close()
    # -print average precision on screen
    print("--> average training time: {}".format(prec))
    # -return average precision
    return (dict, prec)


def collect_all(method_dict, seed_list, args, name=None):
    # -print name of method on screen
    if name is not None:
        print("\n------{}------".format(name))
    # -run method for all random seeds
    for seed in seed_list:
        args.seed = seed
        method_dict[seed] = get_results(args)
        print(method_dict[seed])
    # -return updated dictionary with results
    return method_dict


def get_citation(a):
    if a == 'EWC':
        return '\\cite{EWC} '
    elif a == 'o-EWC':
        return '\\cite{o-EWC} '
    elif a == 'SI':
        return '\\cite{SI} '
    elif a == 'LwF':
        return '\\cite{LwF} '
    elif a == 'GR':
        return '\\cite{GR} '
    elif a == 'GR+distill':
        return '\\cite{DGR} '
    elif a == 'A-GEM':
        return '\\cite{A-GEM} '
    elif a == 'ER':
        return '\\cite{hayes2019memory} '
    elif a == 'iCaRL':
        return '\\cite{iCaRL} '
    else:
        return ''


if __name__ == '__main__':
    args = parser.parse_args()
    form = args.form
    epochs = [1, 5]
    scenarios = ['task', 'domain', 'class']
    algorithms = ['EWC', 'o-EWC', 'SI', 'LwF', 'GR', 'GR+distill', 'A-GEM',
                  'ER', 'iCaRL', 'OTR', 'OTR+distill']

    table_writer = open('./{}_epoch_table_{}.tex'.format(args.experiment, form), 'w+')
    table_writer.write('\\begin{table}[!t]\n')
    table_writer.write('\\renewcommand{\\arraystretch}{1.3}\n')
    table_writer.write('\\caption{Comparison of single-pass and multi-pass over the mini-batch '
                       'on split MNIST dataset}\n')
    table_writer.write('\\label{tab:' + args.experiment.lower() + '_epoch_table}\n')
    table_writer.write('\\centering\n')
    table_writer.write('\\hspace*{-1cm}\\begin{tabular}{l')
    for i in range(len(epochs)):
        table_writer.write('c@{\hskip 0.2cm}c@{\hskip 0.2cm}c@{\hskip 0.2cm}')

    table_writer.write('}\n')
    table_writer.write('\\hline\\hline\n')
    table_writer.write('\\bfseries \celltwoline{Methods} {}	& ')
    for i, ex in enumerate(epochs):
        if i != len(epochs) - 1:
            table_writer.write('\\threecol{' + f'{epochs[i]}-Epoch' + '} &')
        else:
            table_writer.write('\\threecol{' + f'{epochs[i]}-Epoch' + '}\\\\\n')
    table_writer.write('\\cline{2-%d}\n' % (3 * len(epochs) + 1))

    for i in range(len(epochs)):
        if i != len(epochs) - 1:
            table_writer.write(' & Task-IL & Domain-IL & Class-IL ')
        else:
            table_writer.write(' & Task-IL & Domain-IL & Class-IL \\\\\n')
    table_writer.write('\\hline\n')

    for a in algorithms:
        if a == 'OTR' or a == 'OTR+distill':
            table_writer.write(a + " (ours) & ")
        else:
            table_writer.write(a + get_citation(a) + ' & ')
        for ei, e in enumerate(epochs):
            for si, s in enumerate(scenarios):
                args = parser.parse_args()
                args.r_dir = '{}/{}/{}'.format(args.r_dir, args.experiment, s)
                args.scenario = s
                args.epochs = e

                # -set default-values for certain arguments based on chosen scenario & experiment
                args = set_default_values(args)
                # -set other default arguments
                args.lr_gen = args.lr if args.lr_gen is None else args.lr_gen
                args.g_iters = args.iters if args.g_iters is None else args.g_iters
                args.g_fc_lay = args.fc_lay if args.g_fc_lay is None else args.g_fc_lay
                args.g_fc_uni = args.fc_units if args.g_fc_uni is None else args.g_fc_uni

                # Add non-optional input argument that will be the same for all runs
                args.metrics = True
                args.feedback = False
                args.log_per_task = True

                # Add input arguments that will be different for different runs
                args.distill = False
                args.agem = False
                args.ewc = False
                args.online = False
                args.si = False
                args.xdg = False
                args.add_exemplars = False
                args.bce_distill = False
                args.icarl = False
                args.mem_online = False
                seed_list = list(range(args.seed, args.seed + args.n_seeds))

                set_params(args, a)

                DATA = {}
                if a == 'iCaRL':
                    if s == 'class':
                        DATA = collect_all(DATA, seed_list, args, name=a)

                        acc = []
                        for m in seed_list:
                            acc.append(DATA[m][1])

                        mean = np.mean(acc)
                        std = np.std(acc)

                        if ei == len(epochs) - 1 and si == len(scenarios) - 1:
                            if form == 'NIPS':
                                table_writer.write('{:.2f} \\\\\n'.format(mean))
                            else:
                                table_writer.write('{:.2f} ($\pm${:.3f}) \\\\\n'.format(mean, std))
                        else:
                            if form == 'NIPS':
                                table_writer.write('{:.2f} & '.format(mean))
                            else:
                                table_writer.write('{:.2f} ($\pm${:.3f}) & '.format(mean, std))
                    else:
                        table_writer.write('- & ')
                else:
                    DATA = collect_all(DATA, seed_list, args, name=a)
                    acc = []
                    for m in seed_list:
                        acc.append(DATA[m][1])

                    mean = np.mean(acc) * 100
                    std = np.std(acc) * 100

                    if ei == len(epochs) - 1 and si == len(scenarios) - 1:
                        if form == 'NIPS':
                            table_writer.write('{:.2f} \\\\\n'.format(mean))
                        else:
                            table_writer.write('{:.2f} ($\pm${:.3f}) \\\\\n'.format(mean, std))
                    else:
                        if form == 'NIPS':
                            table_writer.write('{:.2f} & '.format(mean))
                        else:
                            table_writer.write('{:.2f} ($\pm${:.3f}) & '.format(mean, std))
                # reset_params(args, a)

        if a == 'Offline':
            table_writer.write('\\hline\n')
    table_writer.write('\\hline\n')
    table_writer.write('\\end{tabular}\n')
    table_writer.write('\\end{table}\n')
    table_writer.close()
