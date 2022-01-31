import torch
from core import NN
from memory import Memory
from torch.autograd import Variable
import random as rd

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
        self.lr = 0.0003

        # Tools
        self.test = False
        self.nbEvents = 0

        # Model and optimisation initialisation
        self.inSize = self.featureExtractor.outSize
        self.outSize = self.env.action_space.n
        self.layers = torch.tensor([200])
        self.model = NN(self.inSize, self.outSize, self.dueling, self.noisy, self.layers)
        self.predNN = self.model.cuda()
        self.targetNN = self.model.cuda() if self.target or self.double else self.predNN
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.predNN.parameters(), lr=self.lr)
        if replay:
            self.memory = Memory(opt.memory_size, prior=True) if self.prioritized else Memory(opt.memory_size, prior=False)


    def act(self, obs):
        # test mode, no exploration
        self.update_explo()
        if self.test or self.noisy:
            self.action = torch.argmax(self.predNN(obs)).cuda()

        # train mode, exploration
        else:
            if rd.uniform(0, 1) < self.explo:
                self.action = self.env.action_space.sample()
            else:
                self.action = torch.argmax(self.predNN(obs)).cuda()

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
                states = torch.stack([transactions[i][0] for i in range(self.batch_size)], dim=0).cuda()
                actions = torch.tensor([transactions[i][1] for i in range(self.batch_size)]).cuda()
                rewards = [transactions[i][2] for i in range(self.batch_size)]
                next_states = torch.stack([transactions[i][3] for i in range(self.batch_size)], dim=0).cuda()
                dones = [transactions[i][4] for i in range(self.batch_size)]

                # Q function of current state
                states = Variable(states).float()
                pred = self.predNN(states).squeeze().gather(-1, actions.unsqueeze(1)).squeeze()

                # Q function of next state
                next_states = Variable(next_states).float().cuda()
                next_pred = self.targetNN(next_states).data
                rewards = torch.FloatTensor(rewards).cuda()

                # Q Learning: get maximum Q value at s' from target model
                dones = list(map(int, dones))
                dones = torch.FloatTensor(dones).cuda()
                if self.double:
                    target = rewards + (1 - dones) * self.discount * torch.gather(next_pred, -1, torch.argmax(self.predNN(next_states), dim=1).unsqueeze(1)).squeeze()
                else :
                    target = rewards + (1 - dones) * self.discount * next_pred.max(1)[0]
                target = Variable(target)

                # Memory update
                if self.prioritized:
                    terr = torch.abs(pred - target).data
                    self.memory.update(idxs, terr)

                # Optimisation
                self.optimizer.zero_grad()
                loss = (torch.FloatTensor(weights).cuda() * self.criterion(pred, target)).mean() if self.prioritized else self.criterion(pred, target)
                loss.backward()

                # and train
                self.optimizer.step()

                if self.noisy:
                    self.predNN.reset_noise()
                    self.targetNN.reset_noise()

            if not self.replay:
                state, action, reward, next_state, done = self.lastTransition
                states = torch.tensor(state).cuda()
                actions = torch.tensor(action).cuda()
                rewards = [reward]
                next_states = torch.tensor(next_state).cuda()
                dones = [done]

                # Q function of current state
                states = Variable(states).float()
                pred = self.predNN(states).squeeze().gather(-1, actions).squeeze()

                # Q function of next state
                next_states = Variable(next_states).float().cuda()
                next_pred = self.targetNN(next_states).data
                rewards = torch.FloatTensor(rewards).cuda()

                # Q Learning: get maximum Q value at s' from target model
                dones = list(map(int, dones))
                dones = torch.FloatTensor(dones).cuda()
                if self.double:
                    target = rewards + (1 - dones) * self.discount * next_pred[torch.argmax(self.predNN(next_states))]
                else :
                    target = rewards + (1 - dones) * self.discount * next_pred.max(-1)[0]
                target = Variable(target)

                # Memory update
                if self.prioritized:
                    terr = torch.abs(pred - target).data
                    self.memory.update(idxs, terr)

                # Optimisation
                self.optimizer.zero_grad()
                loss = (torch.FloatTensor(weights).cuda() * self.criterion(pred, target)).mean() if self.prioritized else self.criterion(pred, target)
                loss.backward()

                # and train
                self.optimizer.step()

                if self.noisy:
                    self.predNN.reset_noise()
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
            self.lastTransition = tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)

    # Dans cette version retourne vrai tous les freqoptim evenements
    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.freqOptim == 0

    def timeToTransfer(self, jsum):
        if self.test:
            return False
        return self.nbEvents % self.freqTransfer == 0