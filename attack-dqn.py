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
from deepfool import DeepFool
from apgdt import APGDT
from apgd import APGD
from difgsm import DIFGSM 
from ffgsm import FFGSM 
from mifgsm import MIFGSM
from pgd import PGD
from gn import GN
from tifgsm import TIFGSM
import argparse
import os
import sys
import time
from strategy import Strategy
from parser import parser
from defendedDQN import CnnDQN
from defendedEnv import atari_env
from defendedUtils import read_config
from SADQNModel import model_setup
from SADQNWrapper import make_atari, wrap_deepmind, wrap_pytorch


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
defended = args.defended

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
if defended == 1:
	setup_json = read_config(args.env_config)
	env_conf = setup_json["Default"]
	for i in setup_json.keys():
					if i in args.env:
							env_conf = setup_json[i]
	env = atari_env(args.env, env_conf, args)
	net = CnnDQN(env.observation_space.shape[0], env.action_space)
	#net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))
	net.load_state_dict(torch.load(model, map_location= torch.device('cpu')))
elif defended == 2:
	env_id = args.env
	env_params =  {"crop_shift": 10,"restrict_actions": 4}
	env_params['clip_rewards'] = False
	env_params['episode_life'] = False
	env = make_atari(env_id)
	env = wrap_deepmind(env, **env_params)
	env = wrap_pytorch(env)
	dueling = True
	robust_model = False
	USE_CUDA = torch.cuda.is_available()
	model_path = args.Path
	model_width = 1
	net = model_setup(env_id, env, robust_model, USE_CUDA, dueling, model_width)
	net.features.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	#action = net.act(state_tensor)[0]
else:
	net = DQN(env.observation_space.shape, env.action_space.n)
	net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

print("Done Loading Model")
Numberofgames = 0
total_reward = 0.0
orig_actions = []
Totalsteps = 0
Allsteps = 0
successes = 0
attack_times = []
adv_actions = []
Total_rewards = []
total_rewardPerEpisode = 0

while Numberofgames != TotalGames:

	state = env.reset()

	while True:
			#attack = True

			start_ts = time.time()
			if visualize:
				env.render()
			state_v = torch.tensor(np.array([state], copy=False))
			if defended == 2:
				state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).to(torch.float32)
				orig_action = net.act(state_tensor)[0]
				print(orig_action)
			else:
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
						domain = True
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
						elif perturbationType == "rfgsm" or perturbationType == "RFGSM" or perturbationType == "Rfgsmt":
							rfgsmIns = RFGSM(model = net, targeted = targeted, steps = stepsRFGSM, eps = epsRFGSM, alpha = alphaRFGSM)
						elif perturbationType == "fgsm" or perturbationType == "FGSM":
							rfgsmIns = FGSM(model = net, targeted = targeted)
						elif perturbationType == "cw" or perturbationType == "CW":
							rfgsmIns = CW(model = net, targeted = targeted)
						elif perturbationType == "apgd" or perturbationType == "APGD":
							rfgsmIns = APGD(model = net, targeted = targeted)
						elif perturbationType == "apgdt" or perturbationType == "APGDT":
							rfgsmIns = APGDT(model = net, targeted = targeted)
						elif perturbationType == "difgsm" or perturbationType == "DIFGSM":
							rfgsmIns = DIFGSM(model = net, targeted = targeted)
						elif perturbationType == "ffgsm" or perturbationType == "FFGSM":
							rfgsmIns = FFGSM(model = net, targeted = targeted)
						elif perturbationType == "mifgsm" or perturbationType == "MIFGSM":
							rfgsmIns = MIFGSM(model = net, targeted = targeted)
						elif perturbationType == "pgd" or perturbationType == "PGD":
							rfgsmIns = PGD(model = net, targeted = targeted)
						elif perturbationType == "gn" or perturbationType == "GN":
							rfgsmIns = GN(model = net, targeted = targeted)
						elif perturbationType == "tifgsm" or perturbationType == "TIFGSM":
							rfgsmIns = TIFGSM(model = net, targeted = targeted)
						elif perturbationType == "gn" or perturbationType == "GN":
							rfgsmIns = GN(model = net)
							

						if strategy == "critical":
							for i in range(len(adv_acts)):
								print("Attacking...")
								adv_act = torch.tensor(np.array([adv_acts[i].item()], copy=False))
								print(adv_act.shape)

								if defended != 2:
									adv_state = rfgsmIns.forward(state_tensor,adv_act)
								else:
									adv_state = rfgsmIns.forward(state_v,adv_act)
								attack_times.append(time.time() - start_attack)
								if defended == 2:
									adv_action = net.act(adv_state)[0]
								else:
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
							if targeted != 0:
								rfgsmIns.set_mode_targeted_by_function(target_map_function=lambda images, labels:labels)
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

			else:
				Allsteps += 1
				orig_actions.append(orig_action)
				state, reward, done, _ = env.step(orig_action)

			total_reward += reward
			total_rewardPerEpisode += reward
			print("Episode Number: ", Numberofgames+1, "/", TotalGames, "	Reward so far Per Episode: ", total_rewardPerEpisode, "	Done status: ", done)
			if done:
				Total_rewards.append(total_rewardPerEpisode)
				total_rewardPerEpisode = 0
				Numberofgames += 1
				break
			if visualize:
				delta = 1/FPS - (time.time() - start_ts)
				if delta > 0:
					time.sleep(delta)

average_reward = total_reward/TotalGames
Total_rewards = np.array(Total_rewards)
average_reward = np.mean(Total_rewards, axis = 0)
std_average_reward = np.std(Total_rewards, axis = 0)
print("Average reward: %.2f" % average_reward)
print("Standard deviation of Rewards: %.2f" % std_average_reward)
print("All Rewards Distribution: %s" % Total_rewards)
print("Predicted DRL agent Actions: ", orig_actions)
if Doattack:
	attack_times = np.array(attack_times)
	average_state_P_time = sum(attack_times)/len(attack_times)
	average_state_P_time = np.mean(attack_times, axis = 0)
	average_state_P_time_std = np.std(attack_times, axis = 0)
	successRate = successes/Totalsteps
	attackRate = Totalsteps/Allsteps
	print("Adversarial Actions: ", adv_actions)
	print("Success rate: %.2f" % successRate)
	print("Total steps Attacked: %f" % Totalsteps)
	print("Attack rate: %f" % attackRate)
	print("Total Attack Execution Time: %.2f seconds" % np.sum(attack_times))
	print("Average One Perturbed state generation time: %f seconds" % average_state_P_time)
	print("STD One Perturbed state generation time: %f seconds" % average_state_P_time_std)
	print("Attack Times List: %s" % attack_times)

print("Overall Program Execution Time: %.2f seconds" % (time.time() - start_time_program))

sys.exit(0)
import os
os._exit(0)
raise SystemExit
