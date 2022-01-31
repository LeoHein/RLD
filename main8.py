import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import yaml
from datetime import datetime
import gym
import gridworld
import torch
import numpy as np

from SACAgent import SACAgent
from utils8 import *
from core8 import *
from memory8 import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', '--env', type=str, required=True, choices=['cartpole', 'mountaincar', 'gridworld', 'lunar', 'pendulum'])
    args = vars(parser.parse_args())

    env, config, outdir, logger = init('./configs/config_random_' + args['env'] + '.yaml', "DQN")

    discount = config['discount']
    lr = config['lr_actor']
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]

    agent = SACAgent(env, config)

    writerPath = "runs/" + str(args['env']) + "/" + str(lr) + "-" + str(discount)
    writer = SummaryWriter(writerPath)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    m=0
    k = 0

    for i in range(episode_count):
        checkConfUpdate(outdir, config)
        rsum = 0
        ob = env.reset()
        if i%10==0:
            print("alpha ", agent.alpha)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:
            print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        j = 0

        new_ob = agent.featureExtractor.getFeatures(ob)
        while True:

            ob = new_ob
            action = agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)

            j += 1
            k += 1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True

            agent.store(ob, action, new_ob, reward, done, j)
            rsum += reward
            agent.nbEvents+=1

            if agent.timeToLearn(i):
                agent.nbEvents=0
                agent.learn()
            if done:
                m+=rsum
                if i%10==0:
                    m=m/10
                    print(str(i) + " Mrsum=" + str(m))
                    m=0
                logger.direct_write("reward", rsum, i)
                if i > 20:
                    logger.direct_write("mean entropy", agent.mean_entropy, i)
                    logger.direct_write("loss Q2", agent.loss_Q, i)
                    logger.direct_write("loss Pi", agent.loss_Pi, i)
                    logger.direct_write("loss alpha", agent.loss_Alpha, i)
                writer.add_scalar("reward", rsum, i)
                mean += rsum
                rsum = 0
                break
    env.close()
