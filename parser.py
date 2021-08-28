import argparse
def parser():

	parser = argparse.ArgumentParser(description = "Fast Undetectable Attack")
	parser.add_argument('-mp','--Path', metavar = 'path', type = str, help = 'Complete path to model')
	parser.add_argument('-e','--env', type = str, nargs = "?", default = "PongNoFrameskip-v4", help = 'Environment name like PongNoFrameskip-v4')
	parser.add_argument('-p','--perturbationType', nargs="?", default="rfgsm", type = str, help = 'Perturbation Type: fgsm, rfgsm, cw, optimal')
	parser.add_argument('-a', '--attack', nargs="?", default=1, type = int, help = 'Attack 1 or not to attack 0')
	parser.add_argument('--stepsRFGSM', nargs = "?", default = 1, type = int, help = "Number of steps of RFGSM attack")
	parser.add_argument('--alphaRFGSM', nargs = "?", default = 8/255, type = float, help = "Alpha (Step Size) of RFGSM attack")
	parser.add_argument('--epsRFGSM', nargs = "?", default = 16/255, type = float, help = "Epsilon (strength) of RFGSM attack")
	parser.add_argument('--totalgames', nargs = "?", default = 10, type = int, help = "total games/episodes")
	parser.add_argument('--strategy', nargs = "?", default = "allSteps", type = str, help = "Attack strategy: random, allSteps, leastSteps, critical")
	parser.add_argument('--targeted', nargs = "?", default = 0, type = int, help = "0 or 1")
	parser.add_argument('--defended', nargs = "?", default = -1, type = int, help = "-1 (non-defended DRL agent) or 1 (RADIAL)")	

	parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')

	parser.add_argument(
	    '--env-config',
	    default='configDefended.json',
	    metavar='EC',
	    help='environment to crop and resize info (default: config.json)')
	parser.add_argument(
	    '--load-path',
	    default='trained_models/PongNoFrameskip-v4_robust.pt',
	    metavar='LMD',
	    help='path to trained model file')
	parser.add_argument(
	    '--gpu-id',
	    type=int,
	    default=-1,
	    help='GPU to use [-1 CPU only] (default: -1)')
	parser.add_argument(
	    '--skip-rate',
	    type=int,
	    default=4,
	    metavar='SR',
	    help='frame skip rate (default: 4)')
	parser.add_argument(
	    '--fgsm-video',
	    type=float,
	    default=None,
	    metavar='FV',
	    help='whether to to produce a video of the agent performing under FGSM attack with given epsilon')
	parser.add_argument(
	    '--pgd-video',
	    type=float,
	    default=None,
	    metavar='PV',
	    help='whether to to produce a video of the agent performing under PGD attack with given epsilon')
	parser.add_argument('--video',
	                    dest='video',
	                    action='store_true',
	                    help = 'saves a video of standard eval run of model')
	parser.add_argument('--fgsm',
	                    dest='fgsm',
	                    action='store_true',
	                    help = 'evaluate against fast gradient sign attack')
	parser.add_argument('--pgd',
	                   dest='pgd',
	                   action='store_true',
	                   help='evaluate against projected gradient descent attack')
	parser.add_argument('--gwc',
	                   dest='gwc',
	                   action='store_true',
	                   help='whether to evaluate worst possible(greedy) outcome under any epsilon bounded attack')
	parser.add_argument('--action-pert',
	                   dest='action_pert',
	                   action='store_true',
	                   help='whether to evaluate performance under action perturbations')
	parser.add_argument('--acr',
	                   dest='acr',
	                   action='store_true',
	                   help='whether to evaluate the action certification rate of an agent')
	parser.add_argument('--nominal',
	                   dest='nominal',
	                   action='store_true',
	                   help='evaluate the agents nominal performance without any adversaries')

	parser.set_defaults(video=False, fgsm=False, pgd=False, gwc=False, action_pert=False, acr=False)


	return parser