import gym_super_mario_bros
import numpy as np
from gym import wrappers, logger
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.layers.convolutional import Convolution2D
from keras import backend as K
import random, h5py, os.path, pickle, traceback, sys

def pad_state(state, const=0):
    """
    This function pads the mario state to be square.
    :param state: RGB state of MarioBros
    :param const: constant to pad with
    :return: (X,X,3) padded along the first dimension state
    """

    pad_length = state.shape[1] - state.shape[0]
    if pad_length > 0:
        state_padded = np.pad(state, pad_width=[(pad_length, 0), (0, 0), (0, 0)], mode='constant', constant_values=const)
    elif pad_length < 0:
        state_padded = np.pad(state, pad_width=[(0, 0), (abs(pad_length), 0), (0, 0)], mode='constant',constant_values=const)
    else: # If square already
        state_padded = state

    return state_padded


def getIfromRGB(rgb):
    red = rgb[:,:,0]
    green = rgb[:,:,1]
    blue = rgb[:,:,2]
    RGBint = (red<<16) + (green<<8) + blue
    return RGBint


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state, reward, done):
        return self.action_space.sample()


class Qnetwork:
    def __init__(self, env, info):

        # Training Parameters
        self.network_info = info["Network"]

        # Learning parameters
        self.learning_rate = self.network_info["learning_rate"]
        self.gamma = self.network_info["gamma"]

        # Model network function
        self.model = self._build_model(env)

        if info['LoadModel']:
            self.load_model(info['LoadModel'])


    def _build_model(self, env):
        input_2D = env.observation_space.shape[:2]
        input_3D = (1,) + input_2D


        model = Sequential()
        model.add(Reshape(input_3D, input_shape=input_2D))
        model.add(Convolution2D(128, (2, 2), strides=(1, 1), padding="same", activation="relu",kernel_initializer="he_uniform"))
        model.add(Convolution2D(64, (2, 2), strides=(1, 1), padding="same", activation="relu",kernel_initializer="he_uniform"))
        model.add(Convolution2D(32, (2, 2), strides=(1, 1), padding="same", activation="relu",kernel_initializer="he_uniform"))
        model.add(Convolution2D(16, (2, 2), strides=(1, 1), padding="same", activation="relu",kernel_initializer="he_uniform"))


        model.add(Flatten())
        model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
        model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
        model.add(Dense(64, activation="relu", kernel_initializer="he_uniform"))
        model.add(Dense(32, activation="relu", kernel_initializer="he_uniform"))
        model.add(Dense(env.action_space.n, activation="linear"))

        model.compile(loss="mean_squared_error", optimizer="Adam")
        model.summary()

        return model

    def save_model(self, filename):
        # serialize model to JSON
        # model_json = self.model.to_json()
        # with open("models/" + filename + ".json", "w+") as json_file:
        #     json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights("models/" + filename + ".h5")
        print("Saved model to disk in model/" + filename)


    def load_model(self, filename):
        # check first if file exists
        if not os.path.isfile("models/" + filename + ".h5"):
            print("Couldn't load model weights from model/" + filename + ". File does not exist. Creating new model.")
            return False

        self.model.load_weights("models/" + filename + ".h5")
        print("Loaded model weights from disk from model/" + filename)


    def getIfromRGB(self, rgb):
        "RGB to integer converter"
        red = rgb[:, :, 0]
        green = rgb[:, :, 1]
        blue = rgb[:, :, 2]
        RGBint = (red << 16) + (green << 8) + blue
        return RGBint

    def _prepro(self, state):
        "Reshape for (1, DIM) as input to Keras"
        state = self.getIfromRGB(state)
        return state.reshape((1,) + state.shape)


    def update(self, History):
        old_state = History.state_memory[-1]
        new_state = History.state_next_memory
        reward = History.reward_memory[-1]
        action = History.action_memory[-1]


        # Update with reward_nfuture steps based on historical records - hope this prevents getting stuck due to a large pole
        action_oldest = History.action_memory[0]
        X_oldest = self._prepro(History.state_memory[0])
        # rewards = [r * (self.gamma**i) for i, r in enumerate(History.reward_memory)]

        # ##### TEMP ATTEMP reward#####
        Dims = np.shape(History.state_memory)
        rewards = np.zeros(Dims[0])
        for i in range(Dims[0]):
            Xi = self._prepro(History.state_memory[i])

            rewards[i] = (self.gamma ** i) * np.max(self.model.predict(Xi))
        rewards[0] = reward
        ######################

        ##### Steven reward fix #####
        Dims = np.shape(History.state_memory)
        rewards = np.zeros(Dims[0])
        for i in range(Dims[0]):
            Xi = self._prepro(History.state_memory[i])

            rewards[i] = (self.gamma ** i) * np.max(self.model.predict(Xi))
        rewards[0] = reward
        ####################

        Q_oldest = self.model.predict(X_oldest)
        Q_target_oldest = np.copy(Q_oldest)

        Q_target_oldest[0][action_oldest] = np.sum(rewards)

        loss_oldest = Q_target_oldest - Q_target_oldest

        Y_oldest = loss_oldest

        self.model.train_on_batch(X_oldest, Y_oldest)



        # This is updating based on n+1
        X = self._prepro(old_state)
        X_next = self._prepro(new_state)

        # Updating
        # Predict next Qtable
        Q_next = self.model.predict(X_next)
        Q_max_next = np.max(Q_next)

        # Current Qtable
        Q = self.model.predict(X)

        Q_target = np.copy(Q)
        Q_target[0][action] = reward + self.gamma * Q_max_next # To make it kinda two-sequenced prediction

        loss = Q_target - Q

        Y = self.learning_rate * loss

        self.model.train_on_batch(X, Y)


    def best_action(self, state):
        "Gets best action based on current state"
        X = self._prepro(state)
        Q = self.model.predict(X)
        return np.argmax(Q)


    def action_probs(self, state):
        X = self._prepro(state)
        Q = self.model.predict(X)
        Q[Q <0] = 0

        return (Q/np.sum(Q))


class Qagent(object):
    """Qagent"""
    def __init__(self, env, info):
        self.info = info
        self.action_space = env.action_space

        self.Qnetwork = Qnetwork(env, info)

        self.start = 1
        self.iter = 0
        self.state_hist = []
        self.action = self.action_space.sample()
        self.eps_decay = self.info["Agent"]["eps_decay"]

        self.History = Memory(info)


    def _update_statelist(self, state):
        """To store previous state en new state. Previous state is associated with the current reward"""
        if self.start:
            self.state_hist.append(state)
            self.state_hist.append(state)
            self.start = 0
        else:
            del self.state_hist[0]
            self.state_hist.append(state)

    def act(self, state, reward, done):
        self._update_statelist(state)
        if self.start: # If first iter then there's no history to learn from
            self.action = self.action_space.sample()
            self.iter += 1
            return self.action

        old_state = self.state_hist[0]
        new_state = self.state_hist[1]

        self.History.append_to_memory(old_state, new_state, self.action, reward)
        self.Qnetwork.update(self.History)


        # Check for greedy
        self.eps = np.exp(-self.eps_decay * self.iter)

        if np.random.uniform(0, 1) < self.eps: # Be greedy
            self.action = self.action_space.sample()

        elif info["Agent"]["policy"] == "hardmax":
            self.action = self.Qnetwork.best_action(new_state)

        elif info["Agent"]["policy"] == "softmax":
            probs = self.Qnetwork.action_probs(state)
            if np.sum(probs[0] > 0) < 1: # Make sure to not get an error when all probabilities are 0
                self.action = self.action_space.sample()
            else:
                self.action = np.random.choice(len(probs[0]), p=probs[0], replace=False)


        self.iter +=1
        return self.action


    def save_model(self, filename):
        self.Qnetwork.save_model(filename)



class Memory:
    def __init__(self, info):

        # Memory info
        self.memory_info = info["Memory"]


        self.state_memory = []
        self.state_next_memory = []
        self.action_memory = []
        self.reward_memory = []

    def _update_memory(self):
        if len(self.state_memory) > self.memory_info["size"]:

            del self.state_memory[0]
            del self.action_memory[0]
            del self.reward_memory[0]

    def append_to_memory(self, state, state_next, action, reward):
        self.state_memory.append(state)
        self.state_next_memory = state_next
        self.action_memory.append(action)
        self.reward_memory.append(reward)

        self._update_memory()



# load a random world / level and return the env
def init_level(info):
    world = random.choice(info['Worlds'])
    lvl = random.choice(info['Levels'])

    env = gym_super_mario_bros.make(info['Game'] + "-" + str(world) + "-" + str(lvl) + "-" + info['Version'])

    outdir = '/tmp/random-agent-results'
    # env = wrappers.Monitor(env, directory=outdir, force=True) # this line disables closing
    env.seed(0)

    return env


# safe deaths and iter
def save_model_params(info, deaths, iter):
    with open("models/" + info['SaveModel'] + "_params", 'wb') as fp:
        pickle.dump([deaths, iter], fp)



# load default params, or from loaded model if defined
def init_params(info, agent):
    if not info['LoadModel'] or not os.path.isfile("models/" + info['SaveModel'] + "_params"):
        return 0, 0, agent

    # read in params from loaded model
    with open ("models/" + info['SaveModel'] + "_params", 'rb') as fp:
        [deaths, iter] = pickle.load(fp)
        agent.iter = iter
        return deaths, iter, agent



# The actual code
N_iters_explore = 200000

info = {
    "Game" : 'SuperMarioBros',
    "Worlds" : [1,2,3],
    "Levels" : [1], #[1,3,4] level 2 is random shit for all worlds, e.g. water world. See readme
    "Version" : "v1",
    "Network": {"learning_rate": 0.6, "gamma": 0.8},
    "Memory": {"size" : 7},
    "Agent": {"type": 1, "eps_decay":  2.0*np.log(10.0)/N_iters_explore,
              "policy": "softmax" #softmax
               },
   "LoadModel" : "model_easy-dark_1-3_1", # False = no loading, filename = loading (e.g. "model_dark_easy_1-5(=worlds)_13(=levels)")
   "SaveModel" : "model_easy-dark_1-3_1", # False= no saving, filename = saving (e.g. "model_dark_easy_1-5(=worlds)_13(=levels)")
}


# load random mario world/level
env = init_level(info)

if info["Agent"]["type"] == 0:
    agent = RandomAgent(env.action_space)
else:
    agent = Qagent(env, info) # For now

episode_count = 100
reward = 0
done = False

# SS params
reward_run = []
deaths, iter, agent = init_params(info, agent)

try:
    while True:
        env = init_level(info)
        state = env.reset()

        while True:
            action = agent.act(state, reward, done)
            state, reward, done, _ = env.step(action)

            if reward < -50:
                print ("------DEAD!------")
                env.close()
                env = init_level(info)
                env.reset()

                reward_run = []
                reward_run.append(0)
                deaths += 1
            else:
                reward_run.append(reward)
            iter += 1
            print("Iter: {0} | Reward: {1} | Current distance: {2} | Deaths: {3}".format(iter,reward, np.sum(reward_run), deaths))

            # Stops the game
            if done:
                env.close()
                break
except KeyboardInterrupt:
    print ("Interrupted by user, shutting down")
except Exception as e:
    traceback.print_exc()
    print ("Unexpected error:", sys.exc_info()[0] , ": ", str(e))
finally:
    # Close the env and write model / result info to disk
    if env:
        env.close()
    if info['SaveModel']:
        agent.save_model(info['SaveModel'])
        save_model_params(info, deaths, iter)
