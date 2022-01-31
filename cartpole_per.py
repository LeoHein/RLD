from core import *
from memory import *

class DQNAgent2(object):
    """The world's simplest agent!"""

    def __init__(self, env, opt, discount, lr, upfreq):
        self.opt=opt
        self.env=env
        self.model=None
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        else:
            self.savnet=NN(self.featureExtractor.getFeatures(self.env.reset()).shape[1]
                           , self.action_space.n, layers=[200], finalActivation=None
                           , activation=torch.tanh,dropout=0.0)
            self.targetnet=NN(self.featureExtractor.getFeatures(self.env.reset()).shape[1]
                       , self.action_space.n, layers=[200], finalActivation=None
                       , activation=torch.tanh,dropout=0.0)
        self.test=False
        self.majcount=0
        self.nblearn=0
        self.freqOptim=upfreq
        self.batch_size=32
        self.majlim=4
        self.nbEvents=0
        self.explo=0.1
        self.decay=.99999
        self.discount=discount
        self.memory=Memory(10000)
        self.lr=lr
        self.loss=torch.nn.SmoothL1Loss()
        self.optimizer=torch.optim.Adam(self.savnet.parameters(), lr=self.lr)

    def act(self, obs):
        self.explo*=self.decay
        if np.random.uniform() < self.explo:
            return self.action_space.sample()
        else:
            return torch.argmax(self.savnet(torch.tensor(obs, dtype=torch.float))).item()

    # sauvegarde du modèle
    def save(self,outputDir):
        torch.save(self.savnet, outputDir)

    # chargement du modèle.
    def load(self,inputDir):
        self.savnet = torch.load(inputDir)

    # apprentissage de l'agent. Dans cette version rien à faire
    def learn(self):
        self.majcount+=1
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            pass
        else:
            batch = self.memory.sample(self.batch_size)[2]
            act = torch.tensor(list(map(lambda x: x[1], batch)), dtype=torch.long)
            ob = torch.tensor(list(map(lambda x: x[0], batch)), dtype=torch.float)
            y=[]
            for sample in batch:
                reward = sample[2]
                newob = sample[3]
                if sample[4]:  # if done
                    y.append(reward)
                else:
                    y.append(reward + self.discount*torch.max(self.targetnet(torch.tensor(newob, dtype=torch.float))).detach())

            nnoutput=self.savnet(ob).squeeze().gather(-1, act.unsqueeze(1)).squeeze()
            l = self.loss(nnoutput, torch.tensor(y, dtype=torch.float)) # On applique la loss uniquement sur Q(obs, act)
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()

        if self.targetnet is not None:
            if self.majcount>self.majlim:
                # print("update weights", "\n", "\n", "\n", "update weights")
                self.majcount=0
                with torch.no_grad():
                    self.targetnet.load_state_dict(self.savnet.state_dict())

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:
            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                done=False
            tr = (ob, action, reward, new_ob, done)
            self.memory.store(tr)

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self, it):
        if self.test:
            return False
        return it > 10 and self.nbEvents>=self.freqOptim


import argparse
import sys
import matplotlib

# matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
import gridworld
import torch
import numpy as np
from utils import *
from core import *
from memory import *
from torch.utils.tensorboard import SummaryWriter
# import highway_env
from matplotlib import pyplot as plt
import yaml
from datetime import datetime

if __name__ == '__main__':
    env, config, outdir, logger = init('configs/config_dqn_cartpole.yaml', "RandomAgent")

    print(env.observation_space.shape)
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n

    discount = 0.99
    lr = 0.0001
    upfreq = 4

    agent = DQNAgent2(env, config, discount=discount, lr=lr, upfreq=upfreq)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    m = 0
    for i in range(episode_count):
        checkConfUpdate(outdir, config)
        rsum = 0
        ob = env.reset()

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = False
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
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
        if verbose:
            env.render()

        new_ob = agent.featureExtractor.getFeatures(ob)
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action = agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)

            j += 1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or (
                    (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True

            agent.store(ob, action, new_ob, reward, done, j)
            rsum += reward
            agent.nbEvents += 1

            if agent.timeToLearn(i):
                agent.nbEvents = 0
                agent.learn()
            if done:
                m += rsum
                if i % 10 == 0:
                    m = m / 10
                    print(str(i) + " Mrsum=" + str(m))
                    m = 0
                logger.direct_write("reward", rsum, i)
                mean += rsum
                rsum = 0
                break

    env.close()