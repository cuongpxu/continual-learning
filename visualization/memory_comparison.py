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
parser.add_argument('--seed', type=int, default=1, help='[first] random seed (for each random-module used)')
parser.add_argument('--n-seeds', type=int, default=10, help='how often to repeat?')
parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results-dir', type=str, default='../benchmark', dest='r_dir', help="default: %(default)s")

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
model_params.add_argument('--use-teacher', type=bool, default=False, help='Using an offline teacher for distill from memory')
# training hyperparameters / initialization
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--iters', type=int, help="# batches to optimize solver")
train_params.add_argument('--lr', type=float, help="learning rate")
train_params.add_argument('--batch', type=int, default=128, help="batch-size")
train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')

# "memory replay" parameters
replay_params = parser.add_argument_group('Replay Parameters')
replay_params.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
replay_params.add_argument('--online-memory-budget', type=int, default=1000, help="how many sample can be stored?")

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
    # -get average precision
    fileName = '{}/prec-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -print average precision on screen
    print("--> average precision: {}".format(ave))
    # -return average precision
    return (dict, ave)


def collect_all(method_dict, mem_list, seed_list, args, name=None):
    # -print name of method on screen
    if name is not None:
        print("\n------{}------".format(name))
    # -run method for all random seeds and mem size
    for b in mem_list:
        if name in ['A-GEM', 'ER', 'iCaRL']:
            args.budget = b
        else:
            args.online_memory_budget = b
        mem_dict = {}
        for s in seed_list:
            args.seed = s
            mem_dict[s] = get_results(args)
        method_dict[b] = mem_dict
    # -return updated dictionary with results
    return method_dict


if __name__ == '__main__':

    # Load input-arguments
    args = parser.parse_args()
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
    args.bce_distill= False
    args.icarl = False
    # args.seed could of course also vary!

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    mem_list = [1000, 2000, 3000, 4000, 5000]
    seed_list = list(range(args.seed, args.seed + args.n_seeds))

    ###----"BASELINES"----###

    # A-GEM
    args.replay = "exemplars"
    args.distill = False
    args.agem = True
    args.otr_exemplars = False
    AGEM = {}
    AGEM = collect_all(AGEM, mem_list, seed_list, args, name="A-GEM")
    args.replay = "none"
    args.agem = False

    # Experience Replay
    args.replay = "exemplars"
    args.otr_exemplars = False
    ER = {}
    ER = collect_all(ER, mem_list, seed_list, args, name="ER")
    args.replay = "none"

    # Online Replay
    args.replay = 'online'
    args.otr_exemplars = False
    args.triplet_selection = 'HP-HN'
    OTR = {}
    OTR = collect_all(OTR, mem_list, seed_list, args, name='OTR (ours)')
    args.replay = 'none'

    # OTR + distill
    args.replay = 'online'
    args.use_teacher = True
    args.otr_exemplars = False
    args.triplet_selection = 'HP-HN'
    OTRDistill = {}
    OTRDistill = collect_all(OTRDistill, mem_list, seed_list, args, name='OTR+distill (ours)')
    args.replay = 'none'
    args.use_teacher = False

    # iCaRL
    if args.scenario == "class":
        args.bce = True
        args.bce_distill = True
        args.use_exemplars = True
        args.add_exemplars = True
        args.herding = True
        args.norm_exemplars = True
        args.otr_exemplars = False
        ICARL = {}
        ICARL = collect_all(ICARL, mem_list, seed_list, args, name="iCaRL")

        args.bce = True
        args.bce_distill = True
        args.use_exemplars = True
        args.add_exemplars = True
        args.herding = False
        args.norm_exemplars = True
        args.otr_exemplars = True
        ICARLOTR = {}
        ICARLOTR = collect_all(ICARLOTR, mem_list, seed_list, args, name="iCaRL+OTR")

    # Drawing line graph between replay using memory methods
    aGEM_mean = []
    ER_mean = []
    OTR_mean = []
    OTRDistill_mean = []
    iCaRL_mean = []
    iCaRL_OTR_mean = []

    aGEM_std = []
    ER_std = []
    OTR_std = []
    OTRDistill_std = []
    iCaRL_std = []
    iCaRL_OTR_std = []

    for m in mem_list:
        acc_aGEM = []
        acc_ER = []
        acc_OTR = []
        acc_OTRDistill = []
        acc_iCaRL = []
        acc_iCaRL_OTR = []

        ## AVERAGE TEST ACCURACY
        for s in seed_list:
            acc_aGEM.append(AGEM[m][s][1])
            acc_ER.append(ER[m][s][1])
            acc_OTR.append(OTR[m][s][1])
            acc_OTRDistill.append(OTRDistill[m][s][1])
            if args.scenario == "class":
                acc_iCaRL.append(ICARL[m][s][1])
                acc_iCaRL_OTR.append(ICARLOTR[m][s][1])

        aGEM_mean.append(np.mean(acc_aGEM) * 100)
        ER_mean.append(np.mean(acc_ER) * 100)
        OTR_mean.append(np.mean(acc_OTR) * 100)
        OTRDistill_mean.append(np.mean(acc_OTRDistill) * 100)
        if args.scenario == "class":
            iCaRL_mean.append(np.mean(acc_iCaRL) * 100)
            iCaRL_OTR_mean.append(np.mean(acc_iCaRL_OTR) * 100)

        aGEM_std.append(np.std(acc_aGEM) * 100)
        ER_std.append(np.std(acc_ER) * 100)
        OTR_std.append(np.std(acc_OTR) * 100)
        OTRDistill_std.append(np.std(acc_OTRDistill) * 100)
        if args.scenario == "class":
            iCaRL_std.append(np.std(acc_iCaRL) * 100)
            iCaRL_OTR_std.append(np.std(acc_iCaRL_OTR) * 100)

    df = pd.DataFrame({'mem': mem_list,
                       'A-GEM': aGEM_mean, 'A-GEM-std': aGEM_std,
                       'ER': ER_mean, 'ER-std': ER_std,
                       'OTR': OTR_mean, 'OTR-std': OTR_std,
                       'OTR+distill': OTRDistill_mean, 'OTR+distill-std': OTRDistill_std})

    plt.errorbar('mem', 'ER', 'ER-std', data=df, marker='s', color='darkblue', ecolor='darkblue')
    plt.errorbar('mem', 'A-GEM', 'A-GEM-std', data=df, marker='o', markerfacecolor='brown', color='brown', ecolor='brown')
    plt.errorbar('mem', 'OTR', 'OTR-std', data=df, marker='*', color='teal', ecolor='teal')
    plt.errorbar('mem', 'OTR+distill', 'OTR+distill-std', data=df, marker='', color='coral', ecolor='coral')
    if args.scenario == 'class':
        df_iCaRL = pd.DataFrame({'mem': mem_list,
                                 'iCaRL': iCaRL_mean, 'iCaRL-std': iCaRL_std,
                                 'iCaRL+OTR': iCaRL_OTR_mean, 'iCaRL+OTR-std': iCaRL_OTR_std})
        plt.errorbar('mem', 'iCaRL', 'iCaRL-std', data=df_iCaRL, marker='h', color='violet', ecolor='violet')
        plt.errorbar('mem', 'iCaRL+OTR', 'iCaRL+OTR-std', data=df_iCaRL, marker='d', color='peru')

    if args.scenario == 'class':
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    # plt.title('{} memory budget comparison ({}-IL)'.format(args.experiment, args.scenario.capitalize()))
    plt.xlabel('Memory budget')
    plt.ylabel('Accuracy')
    plt.xticks(mem_list)

    plot_margin = 0.2
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0, x1, y0 - plot_margin, y1 ))
    plt.tight_layout()
    # plt.show()
    plt.savefig('./{}_{}_mem_comparison.png'.format(args.experiment, args.scenario))
