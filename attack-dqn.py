import gym
import time
import numpy as np
import torch
import collections
import pyvirtualdisplay
from dqn import DQN, ExperienceReplay, Agent
from wrappers import make_env
from rfgsm import RFGSM


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25

_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_ = _display.start()


# Taken (partially) from 
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/03_dqn_play.py

DirectoryPath = '/content/gdrive/MyDrive/testfolder/'
model=DirectoryPath + 'PongNoFrameskip-v4-best.dat'
record_folder="video"  
visualize=True

env = make_env(DEFAULT_ENV_NAME)
if record_folder:
        env = gym.wrappers.Monitor(env, record_folder, force=True)
net = DQN(env.observation_space.shape, env.action_space.n)
net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

state = env.reset()
total_reward = 0.0
steps = 0
successes = 0

while True:
        attack = True
        start_ts = time.time()
        if visualize:
            env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        orig_action = np.argmax(q_vals)
        orig_action_tensor = torch.tensor(np.array([orig_action], copy=False))
        if attack:
                rfgsmIns = RFGSM(model = net)
                adv_state = rfgsmIns.forward(state_v,orig_action_tensor)
                q_vals = net(adv_state).data.numpy()[0]
                adv_action = np.argmax(q_vals)
                state, reward, done, _ = env.step(adv_action)
                steps +=1
                if adv_action != orig_action:
                    successes +=1
        else:
            state, reward, done, _ = env.step(orig_action)

        total_reward += reward
        if done:
            break
        if visualize:
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
print("Total reward: %.2f" % total_reward)
print("Success rate: %.2f" % successes/steps)

if record_folder:
        env.close()