#!/usr/bin/env python3
import argparse
import os
import numpy as np
from param_stamp import get_param_stamp_from_args
import visual_plt
import main
import utils
from param_values import set_default_values


description = 'Compare CL strategies using various metrics on each scenario of permuted or split MNIST.'
parser = argparse.ArgumentParser('./compare_all.py', description=description)
parser.add_argument('--seed', type=int, default=1, help='[first] random seed (for each random-module used)')
parser.add_argument('--n-seeds', type=int, default=1, help='how often to repeat?')
parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help="default: %(default)s")

# expirimental task parameters.
task_params = parser.add_argument_group('Task Parameters')
task_params.add_argument('--experiment', type=str, default='splitMNIST',
                         choices=['rotMNIST', 'permMNIST', 'splitMNIST',
                                  'CIFAR10', 'CIFAR100',
                                  'CUB2011'])
task_params.add_argument('--scenario', type=str, default='task', choices=['task', 'domain', 'class'])
task_params.add_argument('--tasks', type=int, help='number of tasks')

# specify loss functions to be used
loss_params = parser.add_argument_group('Loss Parameters')
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
model_params.add_argument('--teacher_lr', type=float, help='teacher learning rate')
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
model_params.add_argument('--mem_online', type=utils.str_to_bool, default=False, help='icarl using online exemplar mamagement')
model_params.add_argument('--epochs', type=int, default=1)
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
replay_params.add_argument('--triplet-selection', type=str, default='HP-HN-1', help="Triplet selection strategy")
replay_params.add_argument('--otr-exemplars', type=bool, default=False, help="use otr exemplars instead of random")
replay_params.add_argument('--use-embeddings', type=bool, default=False,
                          help="use embeddings space for otr exemplars instead of features space")
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
eval_params.add_argument('--time', type=utils.str_to_bool, default=True, help="keep track of total training time")
eval_params.add_argument('--pdf', action='store_true', help="generate pdfs for individual experiments")
eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
eval_params.add_argument('--prec-n', type=int, default=1024, help="# samples for evaluating solver's precision")
eval_params.add_argument('--sample-n', type=int, default=64, help="# images to show")

shortcut_params = parser.add_argument_group('Shortcut parameters')
shortcut_params.add_argument('--otr', action='store_true', help='online triplet replay')
shortcut_params.add_argument('--otr_distill', action='store_true', help='online triplet replay with distillation')
shortcut_params.add_argument('--otr_distill_kd', action='store_true', help='online triplet replay with distillation')
shortcut_params.add_argument('--icarl', action='store_true', help="icarl shortcut param")


def get_results(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run; if not do so
    if os.path.isfile("{}/dict-{}.pkl".format(args.r_dir, param_stamp)):
        print("{}: already run".format(param_stamp))
    else:
        print("{}: ...running...".format(param_stamp))
        main.run(args)
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


def collect_all(method_dict, seed_list, args, name=None):
    # -print name of method on screen
    if name is not None:
        print("\n------{}------".format(name))
    # -run method for all random seeds
    for seed in seed_list:
        args.seed = seed
        method_dict[seed] = get_results(args)
    # -return updated dictionary with results
    return method_dict


if __name__ == '__main__':
    ## Load input-arguments
    args = parser.parse_args()
    # -set default-values for certain arguments based on chosen scenario & experiment
    args = set_default_values(args)
    # -set other default arguments
    args.lr_gen = args.lr if args.lr_gen is None else args.lr_gen
    args.g_iters = args.iters if args.g_iters is None else args.g_iters
    args.g_fc_lay = args.fc_lay if args.g_fc_lay is None else args.g_fc_lay
    args.g_fc_uni = args.fc_units if args.g_fc_uni is None else args.g_fc_uni

    # -create results-directory if needed
    utils.make_dirs(args.r_dir)
    # -create plots-directory if needed
    utils.make_dirs(args.p_dir)

    ## Add non-optional input argument that will be the same for all runs
    args.metrics = True
    args.feedback = False
    args.log_per_task = True

    ## Add input arguments that will be different for different runs
    args.distill = False
    args.agem = False
    args.ewc = False
    args.online = False
    args.si = False
    args.xdg = False
    args.add_exemplars = False
    args.bce_distill = False
    args.icarl = False
    # args.seed could of course also vary!

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))


    ###----"BASELINES"----###

    ## Offline
    args.replay = "offline"
    OFF = {}
    OFF = collect_all(OFF, seed_list, args, name="Offline")

    ## None
    args.replay = "none"
    NONE = {}
    NONE = collect_all(NONE, seed_list, args, name="None")


    ###----"TASK-SPECIFIC"----####

    ## XdG
    # if args.scenario=="task" and args.experiment in ['splitMNIST', 'permMNIST', 'rotMNIST']:
    #     args.xdg = True
    #     XDG = {}
    #     XDG = collect_all(XDG, seed_list, args, name="XdG")
    #     args.xdg = False


    ###----"REGULARIZATION"----####

    ## EWC
    args.ewc = True
    EWC = {}
    EWC = collect_all(EWC, seed_list, args, name="EWC")

    ## online EWC
    args.online = True
    args.ewc_lambda = args.o_lambda
    OEWC = {}
    OEWC = collect_all(OEWC, seed_list, args, name="Online EWC")
    args.ewc = False
    args.online = False

    ## SI
    args.si = True
    SI = {}
    SI = collect_all(SI, seed_list, args, name="SI")
    args.si = False


    ###----"REPLAY"----###

    ## LwF
    args.replay = "current"
    args.distill = True
    LWF = {}
    LWF = collect_all(LWF, seed_list, args, name="LwF")

    ## GR
    args.replay = "generative"
    args.distill = False
    if args.experiment in ['CIFAR10', 'CIFAR100']:
        args.lr_gen = 0.0003
    RP = {}
    RP = collect_all(RP, seed_list, args, name="GR")

    ## GR+distill
    args.replay = "generative"
    args.distill = True
    if args.experiment in ['CIFAR10', 'CIFAR100']:
        args.lr_gen = 0.0003
    RKD = {}
    RKD = collect_all(RKD, seed_list, args, name="GR+distill")

    ## A-GEM
    args.replay = "exemplars"
    args.distill = False
    args.agem = True
    AGEM = {}
    AGEM = collect_all(AGEM, seed_list, args, name="AGEM (budget = {})".format(args.budget))
    args.replay = "none"
    args.agem = False

    ## Experience Replay
    args.replay = "exemplars"
    args.mem_online = True
    ER = {}
    ER = collect_all(ER, seed_list, args, name="Experience Replay (budget = {})".format(args.budget))
    args.replay = "none"

    ## Online Replay
    args.replay = 'online'
    args.budget = 2000
    args.triplet_selection = 'HP-HN-1'
    args.bce = True
    if args.scenario == 'class':
        args.bce_distill = True
    args.use_embeddings = False
    args.multi_negative = False
    args.add_exemplars = False
    OTR = {}
    OTR = collect_all(OTR, seed_list, args, name='OTR (ours)')
    args.replay = 'none'
    args.bce = False
    args.bce_distill = False
    args.use_embeddings = False
    args.multi_negative = False
    args.add_exemplars = False

    ## OTR + distill
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
    OTRDistill = {}
    OTRDistill = collect_all(OTRDistill, seed_list, args, name='OTR+distill (ours)')
    args.replay = 'none'
    args.use_teacher = False
    args.use_embeddings = False
    args.multi_negative = False
    args.use_augment = False

    ## Online Replay + embed
    # args.replay = 'online'
    # args.online_memory_budget = 2000
    # args.use_embeddings = True
    # OTREmbed = {}
    # OTREmbed = collect_all(OTREmbed, seed_list, args, name='OTR+Embeds (ours)')
    # args.replay = 'none'
    # args.use_embeddings = False

    ###----"EXEMPLARS + REPLAY"----####

    ## iCaRL
    if args.scenario=="class":
        args.bce = True
        args.bce_distill = True
        args.use_exemplars = True
        args.add_exemplars = False
        args.herding = True
        args.norm_exemplars = True
        args.mem_online = True
        ICARL = {}
        ICARL = collect_all(ICARL, seed_list, args, name="iCaRL (budget = {})".format(args.budget))
        args.bce = False
        args.bce_distill = False
        args.use_exemplars = False
        args.add_exemplars = False
        args.herding = False
        args.norm_exemplars = False
        args.mem_online = False

    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- COLLECT RESULTS -----#
    #---------------------------#

    ## For each task & seed, get the test accuracy of the "base model"
    key = 'acc per task{}'.format(' (all classes up to evaluated task)' if args.scenario=="class" else '')
    base = {}
    for i in range(args.tasks):
        base["task {}".format(i+1)] = {}
        for seed in seed_list:
            base["task {}".format(i+1)][seed] = OFF[seed][0][key]['task {}'.format(i+1)][i]

    ave_prec = {}
    prec = {}
    ave_BWT = {}
    ave_FWT = {}
    ave_F = {}
    ave_F2 = {}
    ave_I = {}

    ## For each seed, create list with average metrics
    for seed in seed_list:
        ## AVERAGE TEST ACCURACY
        ave_prec[seed] = [NONE[seed][1], OFF[seed][1], EWC[seed][1], OEWC[seed][1], SI[seed][1], LWF[seed][1],
                          RP[seed][1], RKD[seed][1], AGEM[seed][1], ER[seed][1],
                          OTR[seed][1], OTRDistill[seed][1]]
        if args.scenario == "class":
            ave_prec[seed].append(ICARL[seed][1])
        # -for plot of average accuracy throughout training
        key = "average"
        prec[seed] = [NONE[seed][0][key], OFF[seed][0][key], EWC[seed][0][key], OEWC[seed][0][key], SI[seed][0][key],
                      LWF[seed][0][key], RP[seed][0][key], RKD[seed][0][key], AGEM[seed][0][key], ER[seed][0][key],
                      OTR[seed][0][key], OTRDistill[seed][0][key]]
        if args.scenario == "class":
            prec[seed].append(ICARL[seed][0][key])

        ## BACKWARD TRANSFER (BWT)
        key = 'BWT'
        ave_BWT[seed] = [NONE[seed][0][key], OFF[seed][0][key], EWC[seed][0][key], OEWC[seed][0][key], SI[seed][0][key],
                         LWF[seed][0][key], RP[seed][0][key], RKD[seed][0][key], AGEM[seed][0][key], ER[seed][0][key],
                         OTR[seed][0][key], OTRDistill[seed][0][key]]
        if args.scenario == "class":
            ave_BWT[seed].append(ICARL[seed][0][key])

        ## FORWARD TRANSFER (FWT)
        key = 'FWT'
        ave_FWT[seed] = [NONE[seed][0][key], OFF[seed][0][key], EWC[seed][0][key], OEWC[seed][0][key], SI[seed][0][key],
                         LWF[seed][0][key], RP[seed][0][key], RKD[seed][0][key], AGEM[seed][0][key], ER[seed][0][key],
                         OTR[seed][0][key], OTRDistill[seed][0][key]]

        if args.scenario == "class":
            ave_FWT[seed].append(ICARL[seed][0][key])

        ## FORGETTING (F)
        key = 'F'
        ave_F[seed] = [NONE[seed][0][key], OFF[seed][0][key], EWC[seed][0][key], OEWC[seed][0][key], SI[seed][0][key],
                       LWF[seed][0][key], RP[seed][0][key], RKD[seed][0][key], AGEM[seed][0][key], ER[seed][0][key],
                       OTR[seed][0][key], OTRDistill[seed][0][key]]

        if args.scenario == "class":
            ave_F[seed].append(ICARL[seed][0][key])

        ## INTRANSIGENCE (I)
        key = 'acc per task{}'.format(' (all classes up to evaluated task)' if args.scenario == "class" else '')
        ave_I[seed] = [
            np.mean([(
                OFF[seed][0][key]['task {}'.format(i+1)][i] - NONE[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
            np.mean([(
                OFF[seed][0][key]['task {}'.format(i + 1)][i] - OFF[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
            np.mean([(
                OFF[seed][0][key]['task {}'.format(i + 1)][i] - EWC[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
            np.mean([(
                OFF[seed][0][key]['task {}'.format(i + 1)][i] - OEWC[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
            np.mean([(
                OFF[seed][0][key]['task {}'.format(i + 1)][i] - SI[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
            np.mean([(
                OFF[seed][0][key]['task {}'.format(i + 1)][i] - LWF[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
            np.mean([(
                OFF[seed][0][key]['task {}'.format(i + 1)][i] - RP[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
            np.mean([(
                OFF[seed][0][key]['task {}'.format(i + 1)][i] - RKD[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
            np.mean([(
                OFF[seed][0][key]['task {}'.format(i + 1)][i] - AGEM[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
            np.mean([(
                OFF[seed][0][key]['task {}'.format(i + 1)][i] - ER[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
            np.mean([(
                    OFF[seed][0][key]['task {}'.format(i + 1)][i] - OTR[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
            np.mean([(
                    OFF[seed][0][key]['task {}'.format(i + 1)][i] - OTRDistill[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
        ]

        if args.scenario=="class":
            ave_I[seed].append(np.mean([(
                OFF[seed][0][key]['task {}'.format(i + 1)][i] - ICARL[seed][0][key]['task {}'.format(i+1)][i]
            ) for i in range(1,args.tasks)]))


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "summary-{}{}-{}".format(args.experiment, args.tasks, args.scenario)
    scheme = "{}-incremental learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)

    # select names / colors / ids
    names = ["None", "Offline"]
    colors = ["grey", "black"]
    ids = [0, 1]
    if args.scenario=="task" and False: # ignore XDG
        names.append("XdG")
        colors.append("purple")
        ids.append(12)
    names += ["EWC", "o-EWC", "SI", "LwF", "GR", "GR+distil", "A-GEM (b={})".format(args.budget),
              "ER (b={})".format(args.budget), "OTR (ours)", "OTR+distill (ours)"]
    colors += ["deepskyblue", "blue", "yellowgreen", "goldenrod", "indianred", "red", "darkblue", "brown",
               "teal", "coral"] # "darkviolet", "coral", 'peru'
    ids += [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    if args.scenario == "class":
        names.append("iCaRL (b={})".format(args.budget))
        colors.append("violet")
        ids.append(12)

    # open pdf
    pp = visual_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []


    ########### AVERAGE ACCURACY ###########

    # bar-plot
    means = [np.mean([ave_prec[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list)>1:
        sems = [np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
        cis = [1.96*np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    figure = visual_plt.plot_bar(means, names=names, colors=colors, ylabel="average precision (after all tasks)",
                                 title=title, yerr=cis if len(seed_list)>1 else None, ylim=(0,1))
    figure_list.append(figure)

    # print results to screen
    result_writer = open('{}/{}_{}_summary.txt'.format(args.p_dir, args.experiment, args.scenario), 'w+')
    result_writer.write("#"*60+"\nSUMMARY RESULTS: {}\n".format(title)+"-"*60 + "\n")
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            result_writer.write("{:19s} {:.2f}  (+/- {:.2f}),  n={}\n".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            result_writer.write("{:19s} {:.2f}\n".format(name, 100*means[i]))
        if i==1:
            result_writer.write("-"*60 + "\n")
    result_writer.write("#"*60 + "\n")

    # line-plot
    x_axes = NONE[args.seed][0]["x_task"]
    ave_lines = []
    sem_lines = []
    for id in ids:
        new_ave_line = []
        new_sem_line = []
        for line_id in range(len(prec[args.seed][id])):
            all_entries = [prec[seed][id][line_id] for seed in seed_list]
            new_ave_line.append(np.mean(all_entries))
            if len(seed_list) > 1:
                new_sem_line.append(1.96*np.sqrt(np.var(all_entries)/(len(all_entries)-1)))
        ave_lines.append(new_ave_line)
        sem_lines.append(new_sem_line)
    figure = visual_plt.plot_lines(ave_lines, x_axes=x_axes, line_names=names, colors=colors, title=title,
                                   xlabel="# of tasks", ylabel="Average precision (on tasks seen so far)",
                                   list_with_errors=sem_lines if len(seed_list)>1 else None)
    figure_list.append(figure)


    ########### BWT & FWT ###########

    # bar-plot BWT
    means = [np.mean([ave_BWT[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list) > 1:
        sems = [np.sqrt(np.var([ave_BWT[seed][id] for seed in seed_list]) / (len(seed_list) - 1)) for id in ids]
        cis = [1.96 * np.sqrt(np.var([ave_BWT[seed][id] for seed in seed_list]) / (len(seed_list) - 1)) for id in ids]
    figure = visual_plt.plot_bar(means, names=names, colors=colors, ylabel="Average BWT",
                                 title=title, yerr=cis if len(seed_list) > 1 else None)
    figure_list.append(figure)

    # bar-plot FWT
    means_fwt = [np.mean([ave_FWT[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list) > 1:
        sems_fwt = [np.sqrt(np.var([ave_FWT[seed][id] for seed in seed_list]) / (len(seed_list) - 1)) for id in ids]
        cis_fwt = [1.96 * np.sqrt(np.var([ave_FWT[seed][id] for seed in seed_list]) / (len(seed_list)-1)) for id in ids]
    figure = visual_plt.plot_bar(means_fwt, names=names, colors=colors, ylabel="Average FWT",
                                 title=title, yerr=cis_fwt if len(seed_list) > 1 else None)
    figure_list.append(figure)

    # write results to files
    result_writer.write("#"*49 + "\n")
    result_writer.write(" "*21+"BWT              FWT\n"+"-"*49 + "\n")
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            result_writer.write("{:16s} {:5.2f} ({:.2f})     {:5.2f} ({:.2f})\n".format(
                name, means[i], sems[i], means_fwt[i], sems_fwt[i],
            ))
        else:
            result_writer.write("{:16s}    {:5.2f}            {:5.2f}\n".format(
                name, means[i], means_fwt[i]
            ))
    result_writer.write("#"*49 + "\n")


    ########### Forgetting & Intransigence ###########

    # bar-plot Forgetting
    means = [np.mean([ave_F[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list) > 1:
        sems = [np.sqrt(np.var([ave_F[seed][id] for seed in seed_list]) / (len(seed_list) - 1)) for id in ids]
        cis = [1.96 * np.sqrt(np.var([ave_F[seed][id] for seed in seed_list]) / (len(seed_list) - 1)) for id in ids]
    figure = visual_plt.plot_bar(means, names=names, colors=colors, ylabel="Average Forgetting",
                                 title=title, yerr=cis if len(seed_list) > 1 else None)
    figure_list.append(figure)

    # bar-plot Intransigence
    means_I = [np.mean([ave_I[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list) > 1:
        sems_I = [np.sqrt(np.var([ave_I[seed][id] for seed in seed_list]) / (len(seed_list) - 1)) for id in ids]
        cis_I = [1.96 * np.sqrt(np.var([ave_I[seed][id] for seed in seed_list]) / (len(seed_list) - 1)) for id in ids]
    figure = visual_plt.plot_bar(means_I, names=names, colors=colors, ylabel="Average Intransigence",
                                 title=title, yerr=cis_I if len(seed_list) > 1 else None)
    figure_list.append(figure)

    # print results to screen
    result_writer.write("#" * 49 + "\n")
    result_writer.write(" "*17+" Forgetting      Intransigence\n"+"-"*49 + "\n")
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            result_writer.write("{:16s} {:5.2f} ({:.2f})     {:5.2f} ({:.2f})\n".format(
                name, means[i], sems[i], means_I[i], sems_I[i],
            ))
        else:
            result_writer.write("{:16s}    {:5.2f}            {:5.2f}\n".format(
                name, means[i], means_I[i]
            ))
    result_writer.write("#"*49 + "\n")

    # add all figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    result_writer.write("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))
    result_writer.close()
