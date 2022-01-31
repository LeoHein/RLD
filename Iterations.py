import matplotlib

matplotlib.use("TkAgg")
import gym
import ast
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import random as rd
import matplotlib.pyplot as plt

def Values_Init(statedic):
    LV = [0 for i in range(len(statedic))]
    return LV


def Pol_Init(statedic, mdp):
    Pol = []
    states = [k for (k, val) in statedic.items()]
    for state in states:
        actions = mdp.get(state)
        if actions is not None:
            index = rd.choice(list(actions.keys()))
            Pol.append(index)
        else:
            Pol.append(None)
    return Pol


def Env_Creation(NPlan):
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan" + str(NPlan) + ".txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    statedic, mdp = env.getMDP()
    return env, statedic, mdp


def ArgMax(mdp, statedic, state, LV, gamma, Pol):
    actions = mdp.get(state)
    state_index = statedic.get(state)
    LVa = np.zeros(len(actions))
    for j in range(len(actions)):
        Va = 0
        action = actions.get(j)
        for k in range(len(action)):
            Vs = action[k][1]
            Vindex = statedic.get(Vs)
            V = LV[Vindex]
            Va += action[k][0] * (action[k][2] + gamma * V)
        LVa[j] = Va
    Pol[state_index] = np.argmax(LVa)
    return Pol


def ValueIteration(NPlan, gamma, epsilon):

    # Creation de l'environnement
    env, statedic, mdp = Env_Creation(NPlan)

    # Initialisation de V
    LV = Values_Init(statedic)

    # Value Iteration
    states = [k for (k, val) in mdp.items()]

    i = 0
    old_LV = [10] * len(statedic)
    Pol = [None] * len(statedic)
    LVtemp = LV.copy()

    while np.linalg.norm(np.array(LV)-np.array(old_LV)) >= epsilon:
        old_LV = LV.copy()
        for state in states:
            Vi = -99999999999
            actions = mdp.get(state)
            state_index = statedic.get(state)
            for j in range(len(actions)):
                Via = 0
                action = actions.get(j)
                for k in range(len(action)):
                    Vs = action[k][1]
                    Vindex = statedic.get(Vs)
                    V = LV[Vindex]
                    Via += action[k][0] * (action[k][2] + gamma*V)

                if Via > Vi:
                    Vi = Via

            LVtemp[state_index] = Vi

        LV = LVtemp.copy()
        i += 1

    for state in states:
        Pol = ArgMax(mdp, statedic, state, LV, gamma, Pol)

    return Pol

print(ValueIteration(1, 0.99, 10e-10))


def PolicyIteration(NPlan, gamma, epsilon):

    # Creation de l'environnement
    env, statedic, mdp = Env_Creation(NPlan)

    # Initialisation de V
    LV = Values_Init(statedic)

    # Initialisation de Pol
    Pol = Pol_Init(statedic, mdp)

    i, k = 0, 0
    old_LV = [10] * len(statedic)
    old_Pol = [10] * len(statedic)
    LVtemp = LV.copy()

    states = [k for (k, val) in mdp.items()]

    while old_Pol != Pol:
        while np.linalg.norm(np.array(LV)-np.array(old_LV)) >= epsilon:
            old_LV = LV.copy()
            for state in states:
                state_index = statedic.get(state)
                action = mdp.get(state).get(Pol[state_index])
                Vi = 0
                for a in range(len(action)):
                    Vs = action[a][1]
                    Vindex = statedic.get(Vs)
                    V = LV[Vindex]
                    Vi += action[a][0] * (action[a][2] + gamma*V)
                LVtemp[state_index] = Vi
            LV = LVtemp.copy()
            i += 1

        old_Pol = Pol.copy()
        old_LV = [0] * len(statedic)

        for state in states:
            Pol = ArgMax(mdp, statedic, state, LV, gamma, Pol)
        k += 1
    return Pol

#print(PolicyIteration(0, 0.99, 10e-10))





