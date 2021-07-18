import gym
import time
import numpy as np
import torch
import collections
import pyvirtualdisplay
from dqn import DQN, ExperienceReplay, Agent
from wrappers import make_env
from rfgsm import RFGSM
from fgsm import FGSM
from cw import CW
import argparse
import os
import sys
import time
from strategy import Strategy
from parser import parser

start_time_program = time.time()
args = parser().parse_args()
model = args.Path
DEFAULT_ENV_NAME = args.env #"PongNoFrameskip-v4"
perturbationType = args.perturbationType
stepsRFGSM = args.stepsRFGSM
alphaRFGSM = args.alphaRFGSM
epsRFGSM = args.epsRFGSM
TotalGames = args.totalgames
strategy = args.strategy
targeted = args.targeted

if args.attack == 1:
  Doattack = True
  attack = True
else:
  Doattack = False
  attack = False

print(args)

FPS = 25
_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_ = _display.start()
record_folder="video"  
visualize=True


env = make_env(DEFAULT_ENV_NAME)
if record_folder:
		env = gym.wrappers.Monitor(env, record_folder, force=True)
net = DQN(env.observation_space.shape, env.action_space.n)
net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

Numberofgames = 0
total_reward = 0.0
orig_actions = []
Totalsteps = 0
Allsteps = 0
successes = 0
attack_times = []
adv_actions = []

while Numberofgames != TotalGames:

	state = env.reset()

	while True:
			#attack = True

			start_ts = time.time()
			if visualize:
				env.render()
			state_v = torch.tensor(np.array([state], copy=False))
			q_vals = net(state_v).data.numpy()[0]
			orig_action = np.argmax(q_vals)
			orig_action_tensor = torch.tensor(np.array([orig_action], copy=False))
			if Doattack:
					if strategy == "random":
						strat = Strategy(Totalsteps)
						attack = strat.randomStrategy()

					elif strategy == "leastSteps":
						strat = Strategy(Totalsteps)
						attack = strat.leastStepsStrategy()

					elif strategy == "allSteps":
						attack = True

					elif strategy == "critical":
						strat = Strategy(Totalsteps)
						M = 2
						n = 2
						domain = False
						dam = "pong"
						acts_mask = []
						repeat_adv_act = 1
						fullSearch = False
						delta = 0
						adv_acts, attack = strat.CriticalPointStrategy(M = M, n = n, net = net, state = state, env_orig = env, acts_mask = acts_mask, repeat_adv_act = repeat_adv_act, dam = dam, domain = domain, delta = delta, fullSearch = fullSearch)
						print("Adversarial Acts returned: " ,adv_acts)
						print("Attack Bool: ", attack)

					if attack:
						start_attack = time.time()
						if perturbationType == "optimal":
							rfgsmIns = RFGSM(model = net, steps = stepsRFGSM)
						elif perturbationType == "rfgsmt" or perturbationType == "RFGSMt" or perturbationType == "Rfgsmt":
							rfgsmIns = RFGSM(model = net, targeted = targeted, steps = stepsRFGSM, eps = epsRFGSM, alpha = alphaRFGSM)
						elif perturbationType == "fgsm" or perturbationType == "FGSM":
							rfgsmIns = FGSM(model = net, targeted = targeted)
						elif perturbationType == "cw" or perturbationType == "CW":
							rfgsmIns = CW(model = net, targeted = targeted)

						if strategy == "critical":
							for i in range(len(adv_acts))
								print("Attacking...")
								print(adv_acts[i])
								print(targeted)
								adv_act = torch.tensor(np.array(adv_acts[i], copy=False))
								adv_state = rfgsmIns.forward(state_v,adv_act)
								attack_times.append(time.time() - start_attack)
								q_vals = net(adv_state).data.numpy()[0]
								adv_action = np.argmax(q_vals)
								state, reward, done, _ = env.step(adv_action)
								Totalsteps +=1
								Allsteps += 1
								if adv_action != orig_action:
									orig_actions.append(orig_action)
									adv_actions.append(adv_action)
									successes +=1
								if done:
									print("------done-------")
									attack = False
									break
								state_v = torch.tensor(np.array([state], copy=False))
								q_vals = net(state_v).data.numpy()[0]
								orig_action = np.argmax(q_vals)
								orig_action_tensor = torch.tensor(np.array([orig_action], copy=False))
						
						else: 
							adv_state = rfgsmIns.forward(state_v,orig_action_tensor)						
							attack_times.append(time.time() - start_attack)
							q_vals = net(adv_state).data.numpy()[0]
							adv_action = np.argmax(q_vals)
							state, reward, done, _ = env.step(adv_action)
							Totalsteps +=1
							Allsteps += 1
							if adv_action != orig_action:
								orig_actions.append(orig_action)
								adv_actions.append(adv_action)
								successes +=1
			else:
				Allsteps += 1
				orig_actions.append(orig_action)
				state, reward, done, _ = env.step(orig_action)

			total_reward += reward
			print(total_reward)
			print("Done status: ", done)
			if done:
		
				Numberofgames += 1
				break
			if visualize:
				delta = 1/FPS - (time.time() - start_ts)
				if delta > 0:
					time.sleep(delta)

average_reward = total_reward/TotalGames
print("Average reward: %.2f" % average_reward)
print("Predicted DRL agent Actions: ", orig_actions)
if Doattack:
  #average_state_P_time = sum(attack_times)/len(attack_times)
  successRate = successes/Totalsteps
  attackRate = Totalsteps/Allsteps
  print("Adversarial Actions: ", adv_actions)
  print("Success rate: %.2f" % successRate)
  print("Total steps Attacked: %f" % Totalsteps)
  print("Attack rate: %f" % attackRate)
  print("Total Attack Execution Time: %.2f seconds" % sum(attack_times))
  #print("Average One Perturbed state generation time: %f seconds" % average_state_P_time)
  print("Attack Times List: %s" % attack_times)

print("Overall Program Execution Time: %.2f seconds" % (time.time() - start_time_program))

sys.exit()
