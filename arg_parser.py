from argparse import ArgumentParser

class Arg_parser():
    def __init__(self):   
        parser = ArgumentParser()
        parser.add_argument("--label", default="")
        parser.add_argument("--seed", type=int, default= 0)
        parser.add_argument("--p", type=int, default=97)
        parser.add_argument("--budget", type=int, default= 150000)
        parser.add_argument("--batch_size", type=int, default= 512)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.98)
        parser.add_argument("--weight_decay", type=float, default=0)
        parser.add_argument("--lambda_nuc", type=float, default=0)
        parser.add_argument("--lambda_rank", type=float, default=0.01)
        parser.add_argument("--optimizer", default="Adam")
        parser.add_argument("--starting_point", type=int, default= 1)

        # Grokfast
        parser.add_argument("--filter", type=str, choices=["none", "ma", "ema", "fir", "smoother", "kalman"], default="none")
        parser.add_argument("--alpha", type=float, default=0.99)
        parser.add_argument("--window_size", type=int, default=100)
        parser.add_argument("--lamb", type=float, default=5.0)
        parser.add_argument("--process_noise", type=float, default=1e-4)
        parser.add_argument("--measurement_noise", type=float, default=1e-2)

        # Smoother
        parser.add_argument("--beta", type=float, default=0.98)
        parser.add_argument("--pp", type=float, default=0.01)

        # for MLP
        parser.add_argument("--hidden_dim", type=int, default= 128)
        parser.add_argument("--init_scale", type=float, default=1.0)
        parser.add_argument("--num_epochs", type=int, default= 10000)
        parser.add_argument("--fraction", type=float, default=0.5)
        parser.add_argument("--LoRA_rank", type = int, default= 16)
        parser.add_argument("--switch_epoch", type = int, default= 50)
        parser.add_argument("--init_rank", type = int, default = 5)


        # Ablation studies
        parser.add_argument("--two_stage", action='store_true')
        parser.add_argument("--save_weights", action='store_true')

        # scheduler
        parser.add_argument("--large_lr", type=float, default=1e-2)
        parser.add_argument("--cutoff_steps", type=float, default= 500)
        args = parser.parse_args()


        filter_str = ('_' if args.label != '' else '') + args.filter
        window_size_str = f'_w{args.window_size}'
        alpha_str = f'_a{args.alpha:.3f}'.replace('.', '')
        lamb_str = f'_l{int(args.lamb)}'
        optimizer_str = f'_optimizer{args.optimizer}'
        beta_str = f'_beta{args.beta}'
        pp_str = f'_pp{args.pp}'
        large_lr_str = f'_large_lr{args.large_lr}'
        cutoff_steps_str = f'_cutoff_steps{args.cutoff_steps}'
        
        if args.filter == 'none':
            filter_suffix = ''
        elif args.filter == 'ma':
            filter_suffix = window_size_str + lamb_str
        elif args.filter == 'ema':
            filter_suffix = alpha_str + lamb_str
        elif args.filter == 'smoother':
            filter_suffix = beta_str + pp_str
        elif args.filter == "kalman":
            filter_suffix = (
                f"_p{args.process_noise:.1e}_m{args.measurement_noise:.1e}".replace(".", "")
                + lamb_str
            )
        else:
            raise ValueError(f"Unrecognized filter type {args.filter}")

        optim_suffix = ''
        if args.weight_decay != 0:
            optim_suffix = optim_suffix + f'_wd{args.weight_decay:.1e}'.replace('.', '')
        if args.lr != 1e-3:
            optim_suffix = optim_suffix + f'_lrx{int(args.lr / 1e-3)}'

        two_stage_suffix = '_two_stage' if args.two_stage else ''
        starting_point_suffix = f'_start_at{int(args.starting_point)}'
        nuc_suffix = f'_lambda-nuc{args.lambda_nuc:.2f}'
        rank_suffix = f'_lambda_rank{args.lambda_rank:.2f}'

        args.label = args.label + filter_str + filter_suffix + optim_suffix +  optimizer_str + large_lr_str + cutoff_steps_str
        
        print(f'Experiment results saved under name: {args.label}')
        self.args = args
    
    def return_args(self):
        return self.args