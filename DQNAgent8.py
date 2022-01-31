from core8 import *
from memory8 import *

class DQNAgent(object):

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
            yq=[]
            yv=[]
            for sample in batch:
                reward = sample[2]
                newob = sample[3]
                if sample[4]:  # if done
                    yq.append(reward)
                else:
                    yq.append(reward + self.discount*torch.max(self.targetnet(torch.tensor(newob, dtype=torch.float))).detach())
            print("ob shape ", ob.shape)
            nnoutput=self.savnet(ob).squeeze().gather(-1, act.unsqueeze(1)).squeeze()
            l = self.loss(nnoutput, torch.tensor(yq, dtype=torch.float)) # On applique la loss uniquement sur Q(obs, act)
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
        return it>10 and self.nbEvents>=self.freqOptim
