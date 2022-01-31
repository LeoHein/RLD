import timeit
from utils import *
import numpy as np
from DQNAgent import *
import optuna

def objective(trial):
    # Choose env
    env = 'lunar'

    # Define DQN agent mode
    target = True
    replay = True
    prioritized = False
    double = False
    dueling = False
    noisy = False
    mode = [target, replay, prioritized, double, dueling, noisy]

    # Others
    sectimer = 600
    n = 100
    printbool = True

    # Define DQN agent mode
    target, replay, prioritized, double, dueling, noisy = mode[0], mode[1], mode[2], mode[3], mode[4], mode[5]

    # Start timers
    start = timeit.default_timer()
    import time
    timeout = time.time() + sectimer

    # Get config
    if env == 'lunar':
        env, config, outdir, logger = init('path/config_dqn_lunar.yaml', "RandomAgent")
    if env == 'cartpole':
        env, config, outdir, logger = init('path/content/config_dqn_cartpole.yaml', "RandomAgent")
    if env == 'gridworld':
        env, config, outdir, logger = init('/content/config_dqn_gridworld.yaml', "RandomAgent")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    # Create tuning config
    explo = trial.suggest_float("explo", 0.01, 0.2)
    discount = trial.suggest_float("discount", 0.9, 0.999999)
    decay = trial.suggest_float("decay", 0.9, 0.999999)
    # lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 10, 200)
    freqOptim = trial.suggest_int("freqOptim", 5, 100)

    tuning_config = {"explo": explo,
                     "discount": discount,
                     "decay": decay,
                     # "lr": lr,
                     "batch_size": batch_size,
                     "freqOptim": freqOptim}

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
    while compteur < 10 and i < episode_count and time.time() < timeout:
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
        new_ob = torch.from_numpy(new_ob[0]).float().cuda()

        while True:
            ob = new_ob
            action = agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)
            new_ob = torch.from_numpy(new_ob[0]).float().cuda()

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
                if printbool:
                    print("\n" + str(i + 1) + " mean rsum=" + str(mean_rsum / n))
                trial.report(mean_rsum / n, i + 1)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                mean_rsum_while = mean_rsum / n
                mean_rsum = 0

                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                if mean_rsum_while < 320:
                    compteur = 0
                if mean_rsum_while == 320:
                    compteur += 1
                break

    stop = timeit.default_timer()
    time = stop - start
    print(time)
    print(i)
    print(compteur)
    env.close()
    return mean_rsum_while
