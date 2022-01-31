import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import math
from torch.autograd import Variable

# classe utile pour ajouter du bruit (utile par exemple pour DDPG)
class Orn_Uhlen:
    def __init__(self, n_actions, mu=0, theta=0.15, sigma=0.2):
        self.n_actions = n_actions
        self.X = np.ones(n_actions) * mu
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

    def reset(self):
        self.X = np.ones(self.n_actions) * self.mu

    def sample(self):
        dX = self.theta * (self.mu - self.X)
        dX += self.sigma * np.random.randn(self.n_actions)
        self.X += dX
        return torch.FloatTensor(self.X)


#################### Extracteurs de Features à partir des observations ##################################"

# classe abstraite générique
class FeatureExtractor(object):
    def __init__(self):
        super().__init__()

    def getFeatures(self, obs):
        pass


# Ne fait rien, conserve les observations telles quelles
# A utiliser pour CartPole et LunarLander
class NothingToDo(FeatureExtractor):
    def __init__(self, env):
        super().__init__()
        ob = env.reset()
        ob = ob.reshape(-1)
        self.outSize = len(ob)

    def getFeatures(self, obs):
        # print(obs)
        return obs.reshape(1, -1)


# Ajoute le numero d'iteration (a priori pas vraiment utile et peut destabiliser dans la plupart des cas etudiés)
class AddTime(FeatureExtractor):
    def __init__(self, env):
        super().__init__()
        ob = env.reset()
        ob = ob.reshape(-1)
        self.env = env
        self.maxTime = env.config["duration"]
        self.outSize = len(ob) + 1

    def getFeatures(self, obs):
        # print(obs)
        return np.concatenate((obs.reshape(1, -1), np.array([self.env.steps / self.maxTime]).reshape(1, -1)), axis=1)


######  Pour Gridworld #############################"

class MapFromDumpExtractor(FeatureExtractor):
    def __init__(self, env):
        super().__init__()
        outSize = env.start_grid_map.reshape(1, -1).shape[1]
        self.outSize = outSize

    def getFeatures(self, obs):
        # prs(obs)
        return obs.reshape(1, -1)


# Representation simplifiée, pas besoin d'encoder les murs et les etats terminaux qui ne bougent pas
# Extracteur recommandé pour Gridworld pour la plupart des algos
class MapFromDumpExtractor2(FeatureExtractor):
    def __init__(self, env):
        super().__init__()
        outSize = env.start_grid_map.reshape(1, -1).shape[1]
        self.outSize = outSize * 3

    def getFeatures(self, obs):
        state = np.zeros((3, np.shape(obs)[0], np.shape(obs)[1]))
        state[0] = np.where(obs == 2, 1, state[0])
        state[1] = np.where(obs == 4, 1, state[1])
        state[2] = np.where(obs == 6, 1, state[2])
        return state.reshape(1, -1)


# Representation simplifiée, avec position agent
class MapFromDumpExtractor3(FeatureExtractor):
    def __init__(self, env):
        super().__init__()
        outSize = env.start_grid_map.reshape(1, -1).shape[1]
        self.outSize = outSize * 2 + 2

    def getFeatures(self, obs):
        state = np.zeros((2, np.shape(obs)[0], np.shape(obs)[1]))
        state[0] = np.where(obs == 4, 1, state[0])
        state[1] = np.where(obs == 6, 1, state[1])
        pos = np.where(obs == 2)
        posx = pos[0]
        posy = pos[1]
        return np.concatenate((posx.reshape(1, -1), posy.reshape(1, -1), state.reshape(1, -1)), axis=1)


# Representation (très) simplifiée, uniquement la position de l'agent
# Ne permet pas de gérer les éléments jaunes et roses de GridWorld
class MapFromDumpExtractor4(FeatureExtractor):
    def __init__(self, env):
        super().__init__()
        self.outSize = 2

    def getFeatures(self, obs):
        pos = np.where(obs == 2)
        posx = pos[0]
        posy = pos[1]
        # print(posx,posy)
        return np.concatenate((posx.reshape(1, -1), posy.reshape(1, -1)), axis=1)


# Representation simplifiée, pour conv
class MapFromDumpExtractor5(FeatureExtractor):
    def __init__(self, env):
        super().__init__()

        self.outSize = (3, env.start_grid_map.shape[0], env.start_grid_map.shape[1])

    def getFeatures(self, obs):
        state = np.zeros((1, 3, np.shape(obs)[0], np.shape(obs)[1]))
        state[0, 0] = np.where(obs == 2, 1, state[0, 0])
        state[0, 1] = np.where(obs == 4, 1, state[0, 1])
        state[0, 2] = np.where(obs == 6, 1, state[0, 2])
        return state


# Autre possibilité de représentation, en terme de distances dans la carte
class DistsFromStates(FeatureExtractor):
    def __init__(self, env):
        super().__init__()
        self.outSize = 16

    def getFeatures(self, obs):
        # prs(obs)
        # x=np.loads(obs)
        x = obs
        # print(x)
        astate = list(map(
            lambda x: x[0] if len(x) > 0 else None,
            np.where(x == 2)
        ))
        astate = np.array(astate)
        a3 = np.where(x == 3)
        d3 = np.array([0])
        if len(a3[0]) > 0:
            astate3 = np.concatenate(a3).reshape(2, -1).T
            d3 = np.power(astate - astate3, 2).sum(1).min().reshape(1)

            # d3 = np.array(d3).reshape(1)
        a4 = np.where(x == 4)
        d4 = np.array([0])
        if len(a4[0]) > 0:
            astate4 = np.concatenate(a4).reshape(2, -1).T
            d4 = np.power(astate - astate4, 2).sum(1).min().reshape(1)
            # d4 = np.array(d4)
        a5 = np.where(x == 5)
        d5 = np.array([0])
        # prs(a5)
        if len(a5[0]) > 0:
            astate5 = np.concatenate(a5).reshape(2, -1).T
            d5 = np.power(astate - astate5, 2).sum(1).min().reshape(1)
            # d5 = np.array(d5)
        a6 = np.where(x == 6)
        d6 = np.array([0])
        if len(a6[0]) > 0:
            astate6 = np.concatenate(a6).reshape(2, -1).T
            d6 = np.power(astate - astate6, 2).sum(1).min().reshape(1)
            # d6=np.array(d6)

        # prs("::",d3,d4,d5,d6)
        ret = np.concatenate((d3, d4, d5, d6)).reshape(1, -1)
        ret = np.dot(ret.T, ret)
        return ret.reshape(1, -1)


#######################################################################################


# Si on veut travailler avec des CNNs plutôt que des NN classiques (nécessite une entrée adaptée)
class convMDP(nn.Module):
    def __init__(self, inSize, outSize, layers=[], convs=None, finalActivation=None, batchNorm=False,
                 init_batchNorm=False, activation=torch.tanh, dropout=0.0, maxPool=None):
        super(convMDP, self).__init__()
        # print(inSize,outSize)

        self.inSize = inSize
        self.outSize = outSize
        self.batchNorm = batchNorm
        self.init_batchNorm = init_batchNorm
        self.activation = activation
        self.dropout = None
        if dropout > 0:
            self.dropout = torch.nn.Dropout(dropout)
        print("inSize:", inSize)
        self.convs = None
        if convs is not None:
            self.convs = nn.ModuleList([])
            nbChans, hSize, wSize = inSize
            for x in convs:
                if nbChans != x[0]:
                    raise RuntimeError("incompatible numbers of channels : ", x[0], nbChans)
                self.convs.append(nn.Conv2d(x[0], x[1], x[2], stride=x[3]))
                hSize = ((hSize - x[2]) / x[3]) + 1
                wSize = ((wSize - x[2]) / x[3]) + 1
                nbChans = x[1]
                if maxPool is not None:
                    self.convs.append(nn.MaxPool2d(maxPool))
                    hSize = (hSize - maxPool) + 1
                    wSize = (wSize - maxPool) + 1
                print(nbChans, hSize, wSize)
            inSize = int(nbChans * hSize * wSize)

        # print(inSize)

        self.layers = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        i = 0
        if batchNorm or init_batchNorm:
            self.bn.append(nn.BatchNorm1d(num_features=inSize))
        # print(inSize)
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            if batchNorm:
                self.bn.append(nn.BatchNorm1d(num_features=x))

            # nn.init.xavier_uniform_(self.layers[i].weight)
            nn.init.normal_(self.layers[i].weight.data, 0.0, 0.02)
            nn.init.normal_(self.layers[i].bias.data, 0.0, 0.02)
            i += 1
            inSize = x

        self.layers.append(nn.Linear(inSize, outSize))

        # nn.init.uniform_(self.layers[-1].weight)
        nn.init.normal_(self.layers[-1].weight.data, 0.0, 0.02)
        nn.init.normal_(self.layers[-1].bias.data, 0.0, 0.02)
        self.finalActivation = finalActivation

    def setcuda(self, device):
        self.cuda(device=device)

    def forward(self, x):

        if self.convs is not None:
            x = x.view(-1, *self.inSize)
            n = x.size()[0]
            i = 0
            for c in self.convs:
                x = c(x)
                x = self.activation(x)
                i += 1
            x = x.view(n, -1)

        if self.batchNorm or self.init_batchNorm:
            x = self.bn[0](x)

        x = self.layers[0](x)

        for i in range(1, len(self.layers)):
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
            # if self.drop is not None:
            #    x = nn.drop(x)
            if self.batchNorm:
                x = self.bn[i](x)
            x = self.layers[i](x)

        if self.finalActivation is not None:
            x = self.finalActivation(x)
        # print("f",x.size())
        return x


# Classe basique de NN générique
# Accepte une liste de tailles de couches pour la variables layers (permet de définir la structure)
class NN(nn.Module):
    def __init__(self, inSize, outSize, dueling, noisy, layers=[], finalActivation=None, activation=torch.tanh,
                 dropout=0.0):
        super(NN, self).__init__()

        self.dueling = dueling
        self.noisy = noisy
        self.layers = nn.ModuleList([])

        self.layers.append(nn.Linear(inSize,200))

        if not self.dueling and not self.noisy:
          self.layers.append(nn.Linear(200, outSize))

        if not self.dueling and self.noisy:
          self.layers.append(NoisyLinear(200, outSize))

        self.activation = activation

        if not self.noisy:
            self.advantage1 = nn.Linear(200, 100)
            self.advantage2 = nn.Linear(100, outSize)
            self.value1 = nn.Linear(200, 100)
            self.value2 = nn.Linear(100, 1)

        if self.noisy:
            self.advantage1 = NoisyLinear(200, 100)
            self.advantage2 = NoisyLinear(100, outSize)
            self.value1 = NoisyLinear(200, 100)
            self.value2 = NoisyLinear(100, 1)


    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = self.activation(x)
            x = self.layers[i](x)

        if self.dueling:
            x = self.activation(x)
            advantage = self.advantage1(x)
            advantage = self.activation(advantage)
            advantage = self.advantage2(advantage)
            value = self.value1(x)
            value = self.activation(value)
            value = self.value2(value)
            x = value + advantage - advantage.mean()
        return x

    def setcuda(self, device):
        self.cuda(device=device)

    def reset_noise(self):
      if not self.dueling:
          self.layers[-1].reset_noise()
      self.advantage1.reset_noise()
      self.advantage2.reset_noise()
      self.value1.reset_noise()
      self.value2.reset_noise()


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        if torch.is_tensor(out_features):
            out_features = out_features.item()
        if torch.is_tensor(in_features):
            in_features = in_features.item()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x