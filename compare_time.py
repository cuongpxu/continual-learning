#!/usr/bin/env python3
import argparse
import os
import numpy as np
from param_stamp import get_param_stamp_from_args
import visual_plt
import main
from param_values import set_default_values


description = 'Compare performance & training time of various continual learning methods.'
parser = argparse.ArgumentParser('./compare_time.py', description=description)
parser.add_argument('--seed', type=int, default=1, help='[first] random seed (for each random-module used)')
parser.add_argument('--n-seeds', type=int, default=1, help='how often to repeat?')
parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results-dir', type=str, default='./results_time', dest='r_dir', help="default: %(default)s")

# expirimental task parameters.
task_params = parser.add_argument_group('Task Parameters')
task_params.add_argument('--experiment', type=str, default='splitMNIST',
                         choices=['rotMNIST', 'permMNIST', 'splitMNIST', 'CIFAR10', 'CIFAR100'])
task_params.add_argument('--scenario', type=str, default='task', choices=['task', 'domain', 'class'])
task_params.add_argument('--tasks', type=int, help='number of tasks')

# model architecture parameters
model_params = parser.add_argument_group('Model Parameters')
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
model_params.add_argument('--multi_negative', type=bool, default=False)
# training hyperparameters / initialization
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--iters', type=int, help="# batches to optimize solver")
train_params.add_argument('--lr', type=float, help="learning rate")
train_params.add_argument('--batch', type=int, default=128, help="batch-size")
train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')

# "memory replay" parameters
replay_params = parser.add_argument_group('Replay Parameters')
replay_params.add_argument('--z-dim', type=int, default=100, help='size of latent representation (default: 100)')
replay_params.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")

replay_params.add_argument('--otr_exemplars', action='store_true', help="use otr exemplars instead of random")
replay_params.add_argument('--triplet_selection', type=str, default='HP-HN-1', help="Triplet selection strategy")
replay_params.add_argument('--use_embeddings', action='store_true',
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
cl_params.add_argument('--gamma', type=float, help="--> EWC: forgetting coefficient (for 'online EWC')")
cl_params.add_argument('--fisher-n', type=int, help="--> EWC: sample size estimating Fisher Information")
cl_params.add_argument('--c', type=float, dest="si_c", help="--> SI: regularisation strength")
cl_params.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="--> SI: dampening parameter")
cl_params.add_argument('--gating-prop', type=float, metavar="PROP", help="--> XdG: prop neurons per layer to gate")

# evaluation parameters
eval_params = parser.add_argument_group('Evaluation Parameters')
eval_params.add_argument('--pdf', action='store_true', help="generate pdfs for individual experiments")
eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
eval_params.add_argument('--prec-n', type=int, default=1024, help="# samples for evaluating solver's precision")
eval_params.add_argument('--sample-n', type=int, default=64, help="# images to show")

# shortcut parameters
shortcut_params = parser.add_argument_group('Shortcut parameters')
shortcut_params.add_argument('--otr', action='store_true', help='online triplet replay')
shortcut_params.add_argument('--otr_distill', action='store_true', help='online triplet replay with distillation')
shortcut_params.add_argument('--icarl', action='store_true', help="bce-distill, use-exemplars & add-exemplars")

def get_results(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run; if not do so
    if os.path.isfile("{}/time-{}.txt".format(args.r_dir, param_stamp)):
        print("{}: already run".format(param_stamp))
    else:
        print("{}: ...running...".format(param_stamp))
        main.run(args)
    # -get average precisions & trainig-times
    fileName = '{}/prec-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    fileName = '{}/time-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    training_time = float(file.readline())
    file.close()
    # -print average precision on screen
    print("--> average precision: {}".format(ave))
    # -return tuple with the results
    return (ave, training_time)


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
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    # -create plots-directory if needed
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    ## We need to output text-file with training time and dictionary with metrics
    args.time = True
    args.metrics = False  #--> calculating metrics would take additional time

    ## Add non-optional input argument that will be the same for all runs
    args.agem = False
    args.bce = False
    args.bce_distill = False
    args.icarl = False
    args.use_exemplars = False
    args.add_exemplars = False
    args.budget = 2000
    args.herding = False
    args.norm_exemplars = False
    args.log_per_task = True

    ## As this script runs the comparions in the "RtF-paper" (van de Ven & Tolias, 2018, arXiv),
    ## the empirical Fisher Matrix is used for EWC
    args.emp_fi = True

    ## Add input arguments that will be different for different runs
    args.distill = False
    args.feedback = False
    args.ewc = False
    args.online = False
    args.si = False
    args.xdg = False
    # args.seed could of course also vary!

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))


    ###----"BASELINES"----###

    ## Offline
    args.replay = "offline"
    SO = {}
    SO = collect_all(SO, seed_list, args, name="Offline")

    ## None
    args.replay = "none"
    SN = {}
    SN = collect_all(SN, seed_list, args, name="None")


    ###----"EWC / SI"----####

    ## EWC
    args.ewc = True
    SEWC = {}
    SEWC = collect_all(SEWC, seed_list, args, name="EWC")

    ## online EWC
    args.online = True
    args.ewc_lambda = args.o_lambda
    SOEWC = {}
    SOEWC = collect_all(SOEWC, seed_list, args, name="Online EWC")
    args.ewc = False
    args.online = False

    ## SI
    args.si = True
    SSI = {}
    SSI = collect_all(SSI, seed_list, args, name="SI")
    args.si = False


    ###----"REPLAY"----###

    ## LwF
    args.replay = "current"
    args.distill = True
    SLWF = {}
    SLWF = collect_all(SLWF, seed_list, args, name="LwF")
    args.distill = False

    ## DGR
    args.replay = "generative"
    SRP = {}
    SRP = collect_all(SRP, seed_list, args, name="GR")

    ## DGR+distill
    args.replay = "generative"
    args.distill = True
    SRKD = {}
    SRKD = collect_all(SRKD, seed_list, args, name="GR+distill")

    ## A-GEM
    args.replay = "exemplars"
    args.distill = False
    args.agem = True
    AGEM = {}
    AGEM = collect_all(AGEM, seed_list, args, name="AGEM".format(args.budget))
    args.replay = "none"
    args.agem = False

    ## Experience Replay
    args.replay = "exemplars"
    ER = {}
    ER = collect_all(ER, seed_list, args, name="ER".format(args.budget))
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

    ## iCaRL
    if args.scenario == "class":
        args.bce = True
        args.bce_distill = True
        args.use_exemplars = True
        args.add_exemplars = True
        args.herding = True
        args.norm_exemplars = True
        ICARL = {}
        ICARL = collect_all(ICARL, seed_list, args, name="iCaRL".format(args.budget))

    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- COLLECT RESULTS -----#
    #---------------------------#

    ave_prec = {}
    train_time = {}

    ## Create lists for all extracted <lists> with fixed order
    for seed in seed_list:
        i = 0
        ave_prec[seed] = [
            SO[seed][i], SN[seed][i],
            SEWC[seed][i], SOEWC[seed][i], SSI[seed][i],
            SLWF[seed][i], SRP[seed][i], SRKD[seed][i],
            AGEM[seed][i], ER[seed][i],
            OTR[seed][i], OTRDistill[seed][i]
        ]
        if args.scenario == "class":
            ave_prec[seed].append(ICARL[seed][0])

        i = 1
        train_time[seed] = [
            SO[seed][i], SN[seed][i],
            SEWC[seed][i], SOEWC[seed][i], SSI[seed][i],
            SLWF[seed][i], SRP[seed][i], SRKD[seed][i],
            AGEM[seed][i], ER[seed][i],
            OTR[seed][i], OTRDistill[seed][i]
        ]
        if args.scenario == "class":
            train_time[seed].append(ICARL[seed][1])
    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "summary-{}{}-{}".format(args.experiment, args.tasks, args.scenario)
    scheme = "incremental {} learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)

    # select names / colors / ids
    names = ["None", "Offline"]
    colors = ["grey", "black"]
    ids = [0, 1]
    names += ["EWC", "o-EWC", "SI", "LwF", "GR", "GR+distil", "A-GEM",
              "ER", "OTR (ours)", "OTR+distill (ours)"]
    colors += ["deepskyblue", "blue", "yellowgreen", "goldenrod", "indianred", "red", "darkblue", "brown",
               "teal", "coral"]
    ids += [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    if args.scenario == "class":
        names.append("iCaRL")
        colors.append("violet")
        ids.append(12)

    # open pdf
    pp = visual_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []

    # bar-plot
    means = [np.mean([ave_prec[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list)>1:
        sems = [np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
        cis = [1.96*np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    figure = visual_plt.plot_bar(means, names=names, colors=colors, ylabel="average precision (after all tasks)",
                                 title=title, yerr=cis if len(seed_list)>1 else None, ylim=(0,1))
    figure_list.append(figure)

    # print results to screen
    print("\n\n"+"#"*60+"\nSUMMARY RESULTS: {}\n".format(title)+"-"*60)
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            print("{:12s} {:.2f}  (+/- {:.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            print("{:12s} {:.2f}".format(name, 100*means[i]))
    print("#"*60)

    # scatter-plot (accuracy vs training-time)
    accuracies = []
    times = []
    for id in ids[:-1]:
        accuracies.append([ave_prec[seed][id] for seed in seed_list])
        times.append([train_time[seed][id]/60 for seed in seed_list])
    xmax = np.max(times)
    ylim = (0,1.025)
    figure = visual_plt.plot_scatter_groups(x=times, y=accuracies, colors=colors[:-1], figsize=(12, 15), ylim=ylim,
                                            ylabel="average precision (after all tasks)", names=names[:-1],
                                            xlabel="training time (in min)", title=title, xlim=[0, xmax + 0.05 * xmax])
    figure_list.append(figure)


    # add all figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))
