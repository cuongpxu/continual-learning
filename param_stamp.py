import data
import utils


def get_param_stamp_from_args(args):
    '''To get param-stamp a bit quicker.'''
    from encoder import Classifier
    from vae_models import AutoEncoder

    scenario = args.scenario
    # If Task-IL scenario is chosen with single-headed output layer, set args.scenario to "domain"
    # (but note that when XdG is used, task-identity information is being used so the actual scenario is still Task-IL)
    if args.singlehead and args.scenario == "task":
        scenario = "domain"

    config = data.get_multitask_experiment(
        name=args.experiment, scenario=scenario, tasks=args.tasks, data_dir=args.d_dir, only_config=True,
        verbose=False,
    )

    if args.feedback:
        model = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            fc_layers=args.fc_lay, fc_units=args.fc_units, z_dim=args.z_dim,
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn == "yes" else False, fc_nl=args.fc_nl,
            experiment=args.experiment
        )
        model.lamda_pl = 1.
    else:
        model = Classifier(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            fc_layers=args.fc_lay, fc_units=args.fc_units, fc_drop=args.fc_drop, fc_nl=args.fc_nl,
            fc_bn=True if args.fc_bn == "yes" else False,
            excit_buffer=True if args.xdg and args.gating_prop > 0 else False,
            experiment=args.experiment
        )

    train_gen = True if (args.replay == "generative" and not args.feedback) else False
    if train_gen:
        generator = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'],
            fc_layers=args.g_fc_lay, fc_units=args.g_fc_uni, z_dim=args.g_z_dim, classes=config['classes'],
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn == "yes" else False, fc_nl=args.fc_nl,
            experiment=args.experiment
        )

    model_name = model.name
    replay_model_name = generator.name if train_gen else None
    param_stamp = get_param_stamp(args, model_name, verbose=False, replay=False if (args.replay == "none") else True,
                                  replay_model_name=replay_model_name)
    return param_stamp


def get_param_stamp(args, model_name, verbose=True, replay=False, replay_model_name=None):
    '''Based on the input-arguments, produce a "parameter-stamp".'''

    # -for task
    multi_n_stamp = "{n}-{set}".format(n=args.tasks, set=args.scenario) if hasattr(args, "tasks") else ""
    task_stamp = "{exp}{multi_n}".format(exp=args.experiment, multi_n=multi_n_stamp)
    if verbose:
        print(" --> task:          " + task_stamp)

    # -for model
    model_stamp = model_name
    if verbose:
        print(" --> model:         " + model_stamp)

    # -for hyper-parameters
    hyper_stamp = "{i_e}{num}{epo}-lr{lr}{lrg}-b{bsz}-{optim}".format(
        i_e="e" if args.iters is None else "i", num=args.epochs if args.iters is None else args.iters, lr=args.lr,
        epo='' if args.epochs == 1 else f'-e{args.epochs}',
        lrg=("" if args.lr == args.lr_gen else "-lrG{}".format(args.lr_gen))
        if (hasattr(args, "lr_gen") and args.replay == 'generative') else "",
        bsz=args.batch, optim=args.optimizer,
    )
    if verbose:
        print(" --> hyper-params:  " + hyper_stamp)

    # -for EWC / SI
    if hasattr(args, 'ewc') and ((args.ewc_lambda > 0 and args.ewc) or (args.si_c > 0 and args.si)):
        ewc_stamp = "EWC{l}-{fi}{o}".format(
            l=args.ewc_lambda,
            fi="{}{}".format("N" if args.fisher_n is None else args.fisher_n, "E" if args.emp_fi else ""),
            o="-O{}".format(args.gamma) if args.online else "",
        ) if (args.ewc_lambda > 0 and args.ewc) else ""
        si_stamp = "SI{c}-{eps}".format(c=args.si_c, eps=args.epsilon) if (args.si_c > 0 and args.si) else ""
        both = "--" if (args.ewc_lambda > 0 and args.ewc) and (args.si_c > 0 and args.si) else ""
        if verbose and args.ewc_lambda > 0 and args.ewc:
            print(" --> EWC:           " + ewc_stamp)
        if verbose and args.si_c > 0 and args.si:
            print(" --> SI:            " + si_stamp)
    ewc_stamp = "--{}{}{}".format(ewc_stamp, both, si_stamp) if (
            hasattr(args, 'ewc') and ((args.ewc_lambda > 0 and args.ewc) or (args.si_c > 0 and args.si))
    ) else ""

    # -for XdG
    xdg_stamp = ""
    if (hasattr(args, 'xdg') and args.xdg) and (hasattr(args, "gating_prop") and args.gating_prop > 0):
        xdg_stamp = "--XdG{}".format(args.gating_prop)
        if verbose:
            print(" --> XdG:           " + "gating = {}".format(args.gating_prop))

    # -for replay
    if replay:
        replay_stamp = "{rep}{KD}{agem}{model}{gi}".format(
            rep=args.replay,
            KD="-KD{}".format(args.temp) if args.distill else "",
            agem="-aGEM" if args.agem else "",
            model="" if (replay_model_name is None) else "-{}".format(replay_model_name),
            gi="-gi{}".format(args.gen_iters) if (
                    hasattr(args, "gen_iters") and (replay_model_name is not None) and (
                not args.iters == args.gen_iters)
            ) else ""
        )
        if args.replay == 'online':
            distill = ''
            if hasattr(args, 'use_teacher') and args.use_teacher:
                distill = '-distill-{}{}'.format(args.distill_type,
                                                 '-A' if utils.checkattr(args, 'use_augment') else '')
            teacher_stamp = '{}{}{}{}{}'.format('' if args.teacher_epochs == 100 else f'-e{args.teacher_epochs}',
                                                '' if args.teacher_split == 0.8 else f'-s{args.teacher_split}',
                                                '' if args.teacher_loss == 'CE' else f'-{args.teacher_loss}',
                                                '' if args.teacher_opt == 'Adam' else f'-{args.teacher_opt}',
                                                '' if not args.use_scheduler else '-useSche')

            embeds = '-embeds' if args.use_embeddings else ''
            selection = '' if args.triplet_selection == 'HP-HN-1' else f'({args.triplet_selection})'
            replay_stamp = '{}-b{}{}{}{}{}{}{}'.format(replay_stamp, args.budget,
                                                       '' if not args.multi_negative else 'MN',
                                                       f'{distill}', f'{teacher_stamp}',
                                                       f'{embeds}', f'{selection}',
                                                       '-addEx' if args.add_exemplars else '')
        if verbose:
            print(" --> replay:        " + replay_stamp)
    replay_stamp = "--{}".format(replay_stamp) if replay else ""

    # -for exemplars / iCaRL
    exemplar_stamp = ""
    if args.replay == "exemplars" or (args.add_exemplars or args.use_exemplars) or utils.checkattr(args, 'icarl'):
        exemplar_opts = "b{}{}{}{}".format(args.budget,
                                           "H" if args.herding else "",
                                           "N" if args.norm_exemplars else "",
                                           "-online" if args.mem_online else "")
        use = "{}{}{}".format("addEx-" if args.add_exemplars else "",
                              "useEx-" if args.use_exemplars else "",
                              "OTR-" if args.otr_exemplars else "")
        exemplar_stamp = "--{}{}".format(use, exemplar_opts)
        if verbose:
            print(" --> exemplars:     " + "{}{}".format(use, exemplar_opts))

    # -for binary classification loss
    binLoss_stamp = ""
    # if hasattr(args, 'bce') and args.bce:
    #     if not ((hasattr(args, 'otr') and args.otr) or (hasattr(args, 'otr_distill') and args.otr_distill)):
    #         binLoss_stamp = '--BCE_dist' if (args.bce_distill and args.scenario == "class") else '--BCE'

    # --> combine
    param_stamp = "{}--{}--{}{}{}{}{}{}{}".format(
        task_stamp, model_stamp, hyper_stamp, ewc_stamp, xdg_stamp, replay_stamp, exemplar_stamp,
        binLoss_stamp,
        "-s{}".format(args.seed) if not args.seed == 0 else "",
    )

    ## Print param-stamp on screen and return
    if verbose:
        print(param_stamp)
    return param_stamp
