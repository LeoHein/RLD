import time
import subprocess
from collections import namedtuple,defaultdict
import logging
import json
import os
import threading
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import copy
import random as rd
import yaml
from datetime import datetime
import gym
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import optuna
from optuna.trial import TrialState
import math
import matplotlib.pyplot as plt
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances

from utils import *
from core import *
from memory import *
from gridworld.gridworld_env import *


class DQNAgent(object):

    def __init__(self, env, opt, tuning_opt, target, replay, prioritized, double, dueling, noisy):

        # DNQ mode
        self.target = target
        self.replay = replay
        self.prioritized = prioritized
        self.double = double
        self.dueling = dueling
        self.noisy = noisy

        # Data importation from config
        if opt.fromFile is not None:
            self.load(opt.fromFile)

        # Environment
        self.opt = opt
        self.env = env
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)

        # Hyperparameters
        self.explo = tuning_opt["explo"]
        self.decay = tuning_opt["decay"]
        self.discount = tuning_opt["discount"]
        self.batch_size = tuning_opt["batch_size"]
        self.freqOptim = tuning_opt["freqOptim"]
        self.freqTransfer = tuning_opt["freqTransfer"]
        self.lr = 0.0003

        # Tools
        self.test = False
        self.nbEvents = 0

        # Models
        self.inSize = self.featureExtractor.outSize
        self.outSize = self.env.action_space.n
        self.layers = torch.tensor([200])
        self.predNN = NN(self.inSize, self.outSize, self.dueling, self.noisy, self.layers)
        if self.target or self.double:
            self.targetNN = NN(self.inSize, self.outSize, self.dueling, self.noisy, self.layers)

        # Loss
        self.criterion = torch.nn.SmoothL1Loss()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.predNN.parameters(), lr=self.lr)

        # Memory
        if replay:
            self.memory = Memory(opt.memory_size, prior=True) if self.prioritized else Memory(opt.memory_size,
                                                                                              prior=False)

    def act(self, obs):
        # test mode, no exploration
        self.update_explo()
        if self.test or self.noisy:
            self.action = torch.argmax(self.predNN(obs))

        # train mode, exploration
        else:
            if rd.uniform(0, 1) < self.explo:
                self.action = self.env.action_space.sample()
            else:
                self.action = torch.argmax(self.predNN(obs))

        # Tensor to scalar
        if torch.is_tensor(self.action):
            self.action = self.action.item()

        return self.action

    def learn(self):
        # Mode test : pas d'entraînement
        if self.test:
            pass

        # Mode train
        else:
            # Create minibatch
            if self.replay:
                self.minibatch = self.memory.sample(self.batch_size)
                idxs, weights, transactions = self.minibatch

                # Extract data from transitions
                states = torch.stack([transactions[i][0] for i in range(self.batch_size)], dim=0)
                actions = torch.tensor([transactions[i][1] for i in range(self.batch_size)])
                rewards = [transactions[i][2] for i in range(self.batch_size)]
                next_states = torch.stack([transactions[i][3] for i in range(self.batch_size)], dim=0)
                dones = [transactions[i][4] for i in range(self.batch_size)]

                # Q function of current state
                states = Variable(states).float()
                pred = self.predNN(states).squeeze().gather(-1, actions.unsqueeze(1)).squeeze()

                # Q function of next state
                next_states = Variable(next_states).float()
                if self.target or self.double:
                    next_pred = self.targetNN(next_states).data
                if not self.target and not self.double:
                    next_pred = self.predNN(next_states).data
                rewards = torch.FloatTensor(rewards)

                # Q Learning: get maximum Q value at s' from target model
                dones = list(map(int, dones))
                dones = torch.FloatTensor(dones)
                if self.double:
                    target = rewards + (1 - dones) * self.discount * torch.gather(next_pred, -1,
                                                                                  torch.argmax(self.predNN(next_states),
                                                                                               dim=1).unsqueeze(
                                                                                      1)).squeeze()
                else:
                    target = rewards + (1 - dones) * self.discount * next_pred.max(1)[0]
                target = Variable(target)

                # Memory update
                if self.prioritized:
                    terr = torch.abs(pred - target).data
                    self.memory.update(idxs, terr)

                # Optimisation
                self.optimizer.zero_grad()
                loss = (torch.FloatTensor(weights) * self.criterion(pred,
                                                                           target)).mean() if self.prioritized else self.criterion(
                    pred, target)
                loss.backward()

                # Train
                self.optimizer.step()

                # Noisy
                if self.noisy:
                    self.predNN.reset_noise()
                    if self.target or self.double:
                        self.targetNN.reset_noise()

            if not self.replay:
                state, action, reward, next_state, done = self.lastTransition
                states = torch.tensor(state)
                actions = torch.tensor(action)
                rewards = [reward]
                next_states = torch.tensor(next_state)
                dones = [done]

                # Q function of current state
                states = Variable(states).float()
                pred = self.predNN(states).squeeze().gather(-1, actions).squeeze()

                # Q function of next state
                next_states = Variable(next_states).float()
                if self.target or self.double:
                    next_pred = self.targetNN(next_states).data
                if not self.target and not self.double:
                    next_pred = self.predNN(next_states).data
                rewards = torch.FloatTensor(rewards)

                # Q Learning: get maximum Q value at s' from target model
                dones = list(map(int, dones))
                dones = torch.FloatTensor(dones)
                if self.double:
                    target = rewards + (1 - dones) * self.discount * next_pred[torch.argmax(self.predNN(next_states))]
                else:
                    target = rewards + (1 - dones) * self.discount * next_pred.max(-1)[0]
                target = Variable(target)
                target = target[0]

                # Memory update
                if self.prioritized:
                    terr = torch.abs(pred - target).data
                    self.memory.update(idxs, terr)

                # Optimisation
                self.optimizer.zero_grad()
                loss = (torch.FloatTensor(weights) * self.criterion(pred,
                                                                           target)).mean() if self.prioritized else self.criterion(
                    pred, target)
                loss.backward()

                # Train
                self.optimizer.step()

                # Noisy
                if self.noisy:
                    self.predNN.reset_noise()
                    if self.target or self.double:
                        self.targetNN.reset_noise()

    def transfer(self):
        with torch.no_grad():
            self.targetNN.load_state_dict(self.predNN.state_dict())

    def update_explo(self):
        self.explo = self.explo * self.decay

    # sauvegarde du modèle
    def save(self, outputDir):
        pass

    # chargement du modèle.
    def load(self, inputDir):
        pass

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self, ob, action, new_ob, reward, done, it):
        # test mode, no transition save
        if not self.test:
            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                done = False
            tr = (ob, action, reward, new_ob, done)
            self.lastTransition = tr  # ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)

    # Dans cette version retourne vrai tous les freqoptim evenements
    def timeToLearn(self, done):
        if self.test:
            return False
        self.nbEvents += 1
        return self.nbEvents % self.freqOptim == 0

    def timeToTransfer(self, jsum):
        if self.test:
            return False
        return self.nbEvents % self.freqTransfer == 0

def objective(trial):

    # Start timers
    start = timeit.default_timer()
    import time
    timeout = time.time() + sectimer

    # Get config
    if envir == 'lunar':
        env, config, outdir, logger = init('/content/config_dqn_lunar.yaml', "RandomAgent")
    if envir == 'cartpole':
        env, config, outdir, logger = init('/content/config_dqn_cartpole.yaml', "RandomAgent")
    if envir == 'gridworld':
        env, config, outdir, logger = init('/content/config_dqn_gridworld.yaml', "RandomAgent")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])

    # Create tuning config
    explo = trial.suggest_float("explo", Cexplo[0], Cexplo[1])
    discount = trial.suggest_float("discount", Cdiscount[0], Cdiscount[1])
    decay = trial.suggest_float("decay", Cdecay[0], Cdecay[1])
    batch_size = trial.suggest_int("batch_size", Cbatch_size[0], Cbatch_size[1])
    freqOptim = trial.suggest_int("freqOptim", CfreqOptim[0], CfreqOptim[1])
    freqTransfer = trial.suggest_int("freqTransfer", CfreqTransfer[0], CfreqTransfer[1])

    tuning_config = {"explo": explo,
                     "discount": discount,
                     "decay": decay,
                     "batch_size": batch_size,
                     "freqOptim": freqOptim,
                     "freqTransfer": freqTransfer
                     }

    # Agent
    agent = DQNAgent(env, config, tuning_config, target, replay, prioritized, double, dueling, noisy)

    # Counts and bools
    jsum = 0
    mean = 0
    itest = 0
    reward = 0
    verbose = True
    done = False
    mean_rsum = 0
    mean_rsum_while = 0

    # Iteration over episodes
    i = -1
    compteur = 0
    while compteur < nb_compteur and i < episode_count_optim and time.time() < timeout:
        i += 1
        rsum = 0
        agent.nbEvents = 0
        ob = env.reset()

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            itest += 1
            agent.test = False

        j = 0
        new_ob = agent.featureExtractor.getFeatures(ob)
        new_ob = torch.from_numpy(new_ob[0]).float()

        while True:
            ob = new_ob
            action = agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)
            new_ob = torch.from_numpy(new_ob[0]).float()

            jsum += 1
            j += 1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or (
                    (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True

            agent.store(ob, action, new_ob, reward, done, j)
            if replay:
                agent.memory.store(agent.lastTransition)
            rsum += reward

            if agent.timeToLearn(done):
                agent.learn()

            if agent.target and agent.timeToTransfer(jsum):
                agent.transfer()

            if done:
                mean_rsum += rsum
                if (i + 1) % n == 0:
                    #if (i+1) % 100 == 0:
                        #print("\n" + str(i + 1) + " mean rsum=" + str(mean_rsum / n))
                    trial.report(mean_rsum / n, i + 1)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                    mean_rsum_while = mean_rsum / n
                    if mean_rsum_while < 320:
                        compteur = 0
                    if mean_rsum_while == 320:
                        compteur += 1
                    mean_rsum = 0

                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                break

    stop = timeit.default_timer()
    time = stop - start
    print("time: " + str(time))
    print("iterations: " + str(i))
    print("compteur: " + str(compteur))
    env.close()
    return mean_rsum_while


#=======================================================================================================================
# Environnement
envir = "lunar"

# Define DQN agent mode
# target, replay, prioritized, double, dueling, noisy

mode1 = (False, False, False, False, False, False)
mode2 = (False, True, False, False, False, False)
mode3 = (True, True, False, False, False, False)
mode4 = (False, True, True, False, False, False)
mode5 = (True, True, True, False, False, False)
mode6 = (False, True, False, True, False, False)
mode7 = (False, True, False, False, True, False)
mode8 = (False, True, False, False, False, True)
mode9 = (False, True, True, True, True, True)
#modes = [mode4, mode5, mode3, mode1, mode2, mode6, mode7, mode8, mode9]
modes = [mode9]

# Tune objective function
sectimer = 60 * 10
n = 100
nb_trials = 50
nb_compteur = 10
episode_count_optim = 10000

# Tune evaluation step
episode_count_test = 10000
nb_compteur_test = 10
print_iter = 1

# Tuning config
Cexplo = [0, 0]
Cdiscount = [0.9, 0.999999]
Cdecay = [0.9, 0.999999]
Cbatch_size = [10, 200]
CfreqOptim = [1, 20]
CfreqTransfer = [0, 0]

# Choose if optim and clean steps
do_optim = False
do_clean = False

# Default tuning config if no optim
tuning_config = {}
tuning_config["explo"] = 0.1
tuning_config["discount"] = 0.999
tuning_config["decay"] = 0.99
tuning_config["batch_size"] = 100
tuning_config["freqOptim"] = 5
tuning_config["freqTransfer"] = 10
#=======================================================================================================================


# Define DQN agent mode
Ltrials = []
Lstudies = []
for target, replay, prioritized, double, dueling, noisy in modes:

    print(target, replay, prioritized, double, dueling, noisy)

    if do_optim:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=nb_trials, timeout=None)
        trial = study.best_trials[0]
        Ltrials.append(trial)
        Lstudies.append(study)
        tuning_config = trial.params

    if do_clean:
        import shutil

        shutil.rmtree('/content/XP')

    nb_iter = print_iter
    for k in range(nb_iter):

        # Start timer
        start = timeit.default_timer()
        import time

        timeout = time.time() + sectimer

        env, config, outdir, logger = init('/Users/Leo/Desktop/3A/m2a/RLD/TMEs/TMEenv/configs/config_dqn_lunar.yaml', "RandomAgent")

        freqTest = config["freqTest"]
        freqSave = config["freqSave"]
        nbTest = config["nbTest"]
        env.seed(config["seed"])
        np.random.seed(config["seed"])

        # Agent
        agent = DQNAgent(env, config, tuning_config, target, replay, prioritized, double, dueling, noisy)

        writer = SummaryWriter(outdir)

        # Counts and bools
        jsum = 0
        mean = 0
        itest = 0
        reward = 0
        verbose = True
        done = False
        mean_rsum = 0
        mean_rsum_while = 0
        compteur = 0

        # Iteration over episodes
        for i in tqdm(range(episode_count_test)):
            # for i in range(episode_count_test):
            if i >= episode_count_test or compteur >= nb_compteur_test or time.time() > timeout:
                break
            i += 1
            # checkConfUpdate(outdir, config)

            rsum = 0
            agent.nbEvents = 0
            ob = env.reset()

            # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
            if i % int(config["freqVerbose"]) == 0:
                verbose = True
            else:
                verbose = False

            # C'est le moment de tester l'agent
            if i % freqTest == 0 and i >= freqTest:
                mean = 0
                agent.test = True

            # On a fini cette session de test
            if i % freqTest == nbTest and i > freqTest:
                itest += 1
                logger.direct_write("rewardTest", mean / nbTest, itest)
                agent.test = False

            # C'est le moment de sauver le modèle
            if i % freqSave == 0:
                agent.save(outdir + "/save_" + str(i))

            j = 0
            # if verbose:
            # env.render()

            new_ob = agent.featureExtractor.getFeatures(ob)
            new_ob = torch.from_numpy(new_ob[0]).float()

            while True:
                # if verbose:
                # env.render()

                ob = new_ob
                action = agent.act(ob)
                new_ob, reward, done, _ = env.step(action)
                new_ob = agent.featureExtractor.getFeatures(new_ob)
                new_ob = torch.from_numpy(new_ob[0]).float()

                jsum += 1
                j += 1

                # Si on a atteint la longueur max définie dans le fichier de config
                if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or (
                        (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                    done = True

                agent.store(ob, action, new_ob, reward, done, j)
                if replay:
                    agent.memory.store(agent.lastTransition)
                rsum += reward

                if agent.timeToLearn(done):
                    agent.learn()

                if target and agent.timeToTransfer(jsum):
                    agent.transfer()

                if done:
                    mean_rsum += rsum
                    n = 100
                    if (i + 1) % n == 0:
                        # print("\n" + str(i+1) + " mean rsum=" + str(mean_rsum/n))
                        mean_rsum_while = mean_rsum / n
                        if mean_rsum_while < 320:
                            compteur = 0
                        if mean_rsum_while > 320:
                            compteur += 1
                            # print(compteur)
                        mean_rsum = 0
                    # print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")

                    writer.add_scalar("reward", rsum, i)
                    agent.nbEvents = 0
                    mean += rsum
                    rsum = 0
                    break

        stop = timeit.default_timer()
        time = stop - start
        print('Time: ', time)

env.close()