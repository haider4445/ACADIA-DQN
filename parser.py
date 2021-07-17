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

return parser