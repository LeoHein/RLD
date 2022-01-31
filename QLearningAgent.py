import matplotlib
matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from datetime import datetime
import os
from utils import *
import random as rd


class QLearning(object):
    def __init__(self, env, opt):
        self.opt=opt
        self.action_space=env.action_space
        self.env=env
        self.discount=opt.gamma
        self.alpha=opt.learningRate
        self.explo=opt.explo
        self.decay=opt.decay
        self.exploMode=opt.exploMode #0: epsilon greedy, 1: ucb
        self.sarsa=opt.sarsa
        self.modelSamples=opt.nbModelSamples
        self.test=False
        self.model={}
        self.qstates={}  # dictionnaire d'états rencontrés
        self.values=[]   # contient, pour chaque numéro d'état, les qvaleurs des self.action_space.n actions possibles


    def save(self,file):
       pass


    # enregistre cette observation dans la liste des états rencontrés si pas déjà présente
    # retourne l'identifiant associé à cet état
    def storeState(self, obs):
        observation = obs.dumps()
        s = str(observation)
        ss = self.qstates.get(s, -1)

        # Si l'etat jamais rencontré
        if ss < 0:
            #env.render(mode='human')
            ss = len(self.values)
            self.qstates[s] = ss
            self.values.append(np.ones(self.action_space.n) * 1.0) # Optimism faced to uncertainty (on commence avec des valeurs à 1 pour favoriser l'exploration)
        return ss


    def store(self, ob, action, new_ob, reward, done, it):
        if self.test:
            return
        self.last_source=ob
        self.last_action=action
        self.last_dest=new_ob
        self.last_reward=reward
        if it == self.opt.maxLengthTrain:   # si on a atteint la taille limite, ce n'est pas un vrai done de l'environnement
            done = False
        self.last_done=done


    def act(self, obs, greedy):
        if self.test:
            self.action = np.argmax(self.values[obs])
        else:
            if greedy:
                if rd.uniform(0, 1) < self.explo:
                    self.action = env.action_space.sample()
                else:
                    self.action = np.argmax(self.values[obs])
            else:
                self.action = np.argmax(self.values[obs])
        return self.action


    def update_explo(self):
        self.explo = self.explo * self.decay


    def learn(self):
        # Q-learning
        if self.exploMode == 0:
            self.new_action = self.act(self.last_dest, False)
        # Sarsa
        if self.exploMode == 1:
            self.new_action = self.act(self.last_dest, True)

        ls = int(self.last_source)
        la = int(self.last_action)
        ld = int(self.last_dest)
        na = int(self.new_action)
        lr = float(self.last_reward)
        a = float(self.alpha)
        dc = float(self.discount)

        self.values[ls][la] = self.values[ls][la] + a * (lr + dc * self.values[ld][na] - self.values[ls][la])


    def update_model(self):
        if self.last_source not in self.model.keys():
            self.model[self.last_source] = {}
        self.model[self.last_source][self.last_action] = (self.last_reward, self.last_dest)


    def dyna(self):
        for _ in range(self.modelSamples):
            # randomly choose an state
            _state = list(self.model)[rd.choice(range(len(self.model.keys())))]
            # randomly choose an action
            _action = list(self.model[_state])[rd.choice(range(len(self.model[_state].keys())))]
            # apply model
            _reward, _nxtState = self.model[_state][_action]
            # update Q Values
            self.values[_state][_action] += self.alpha * (_reward + self.discount*np.max(self.values[_nxtState]) - self.values[_state][_action])


    def print_model(self):
        print(self.model)


if __name__ == '__main__':
    env,config,outdir,logger=init('./configs/config_qlearning_gridworld.yaml',"QLearning")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = QLearning(env, config)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    nb = 0

    for i in range(episode_count):
        checkConfUpdate(outdir, config)  # permet de changer la config en cours de run

        rsum = 0
        agent.nbEvents = 0
        ob = env.reset()

        if i % freqTest == 0 and i >= freqTest:  ##### Si agent.test alors retirer l'exploration
            #print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            #print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        new_ob = agent.storeState(ob)

        while True:
            ob = new_ob
            action = agent.act(ob, True)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.storeState(new_ob)

            j+=1

            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                #print("forced done!")

            agent.store(ob, action, new_ob, reward, done, j)
            rsum += reward

            agent.learn()
            #agent.update_model()
            #agent.dyna()



            if done:
                agent.update_explo()
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                break
    agent.print_model()
    env.close()