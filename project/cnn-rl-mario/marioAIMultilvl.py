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
import random, h5py, os.path, pickle, traceback, sys, copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from scipy.misc import imresize
import matplotlib.pyplot as plt


def calc_reward(old_reward, done):
    "Calculates new reward not only based on distance"
    if done and old_reward <= -50: # If dead
        new_reward = -3
    elif done: # If level finished
        new_reward = 2
    elif old_reward >= 8: # If speed up fast
        new_reward = 1
    elif old_reward > 0: # If move to right
        new_reward = 0.5
    elif old_reward == 0: # If stand still slight punishment BUG walking left against the wall also counts as standing still
        new_reward = -0.1
    elif old_reward < 0 and old_reward >= -8: # If move to left
        new_reward = -1
    elif old_reward <-8 and old_reward > -50: # If move to left fast
        new_reward = -1.5
    else: # Also dead?
        new_reward = -3

    return new_reward


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
        self.replay_info = info["Replay"]

        # Learning parameters
        self.learning_rate = self.network_info["learning_rate"]
        self.gamma = self.network_info["gamma"]

        # Model network function
        self.model = self._build_model(env)

        if info['LoadModel']:
            self.load_model(info['LoadModel'])

        # Extra
        self.best_Qvalue = 0


    def _build_model(self, env):
        # input_2D = env.observation_space.shape[:2]
        input_2D = (84, 84)
        input_3D = (1,) + input_2D

        # SS model
        # model = Sequential()
        # model.add(Reshape(input_3D, input_shape=input_2D))
        # model.add(Convolution2D(128, (2, 2), strides=(1, 1), padding="same", activation="relu",kernel_initializer="he_uniform"))
        # model.add(Convolution2D(64, (2, 2), strides=(1, 1), padding="same", activation="relu",kernel_initializer="he_uniform"))
        # model.add(Convolution2D(32, (2, 2), strides=(1, 1), padding="same", activation="relu",kernel_initializer="he_uniform"))
        # model.add(Convolution2D(16, (2, 2), strides=(1, 1), padding="same", activation="relu",kernel_initializer="he_uniform"))
        #
        # model.add(Flatten())
        # model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
        # model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
        # model.add(Dense(64, activation="relu", kernel_initializer="he_uniform"))
        # model.add(Dense(32, activation="relu", kernel_initializer="he_uniform"))
        # model.add(Dense(env.action_space.n, activation="linear"))


        # Model from http://cs229.stanford.edu/proj2016/report/klein-autonomousmariowithdeepreinforcementlearning-report.pdf
        model = Sequential()
        model.add(Reshape(input_3D, input_shape=input_2D))
        model.add(Convolution2D(32, (8, 8), strides=(4, 4), padding="same", activation="relu",kernel_initializer="he_uniform"))
        model.add(Convolution2D(64, (4, 4), strides=(2, 2), padding="same", activation="relu",kernel_initializer="he_uniform"))
        model.add(Convolution2D(128, (3, 3), strides=(1, 1), padding="same", activation="relu",kernel_initializer="he_uniform"))

        model.add(Flatten())
        model.add(Dense(512, activation="relu", kernel_initializer="he_uniform"))
        model.add(Dense(env.action_space.n, kernel_initializer="he_uniform", activation="linear"))


        # Model from original Deep Q learning paper https://arxiv.org/pdf/1312.5602v1.pdf
        # model = Sequential()
        # model.add(Reshape(input_3D, input_shape=input_2D))
        # model.add(Convolution2D(16, (8, 8), strides=(4, 4), padding="same", activation="relu",kernel_initializer="he_uniform"))
        # model.add(Convolution2D(32, (4, 4), strides=(2, 2), padding="same", activation="relu",kernel_initializer="he_uniform"))
        #
        # model.add(Flatten())
        # model.add(Dense(256, activation="relu", kernel_initializer="he_uniform"))
        # model.add(Dense(env.action_space.n, activation="linear"))

        Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss="mean_squared_error", optimizer=Adam)
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


    def _get_reward(self, statelist, reward_first):
        "Calculates the reward based on however many states are in the list. So if there are 2 states it looks two in the future, etc.."

        Dims = np.shape(statelist)

        Qs = self.model.predict(np.squeeze(statelist))
        rewards = np.max(Qs, axis=1)
        rewards[0] = reward_first
        rewards = [(self.gamma ** i) * rewards[i] for i in range(Dims[0])]


        return np.sum(rewards)

    def _randbatch(self, History):
        future_look_size = History.memory_info["size"]
        n_states_stored = History.cur_size

        if n_states_stored <= future_look_size:
            states = History.state_memory
            reward_first = History.reward_memory[0]
            action_first = History.action_memorykernel_initializer="he_uniform"[0]
        else:
            idx = np.random.randint(n_states_stored - future_look_size)
            states = History.state_memory[idx:(idx + future_look_size)]
            reward_first = History.reward_memory[idx]
            action_first = History.action_memory[idx]

        return states, reward_first, action_first



    def update(self, History):

        ### Replay
        # Creates n batches of size n_future_look and fits model
        # The main part that makes this slow is the actual fitting! (not the loopy loops)
        n_replay = self.replay_info["batchsize"]
        if History.cur_size > History.memory_info["size"]:
            X= []
            Y =[]
            for i in range(n_replay):
                states, reward_first, action_first = self._randbatch(History)

                state_0 = states[0]
                reward_0 = self._get_reward(states, reward_first)

                Q = self.model.predict(state_0)

                Q_target = np.copy(Q)
                Q_target[0][action_first] = reward_0

                loss_0 = self.learning_rate * (Q_target - Q)
                if loss_0[0][action_first] >= 0 and loss_0[0][action_first] <= 1:
                    loss_0[0][action_first] = 1




                Y.append(loss_0)
                X.append(state_0)

            X = np.squeeze(np.asarray(X))
            Y = np.squeeze(np.asarray(Y))

            self.model.fit(X, Y, batch_size = self.replay_info["batchsize"], verbose=0, epochs=1)


        ## Normal updates. Looking 1 in future
        X = History.state_memory[-1]
        X_next = History.state_next_memory[-1]
        action = History.action_memory[-1]
        X_next = np.reshape(X_next, (1,) + np.shape(X_next))

        Q_next = self.model.predict(X_next)
        Q_max_next = np.max(Q_next)
        Q = self.model.predict(X)

        Q_target = np.copy(Q)
        Q_target[0][action] = reward + self.gamma * Q_max_next # To make it kinda two-sequenced prediction

        loss = Q_target - Q
        if loss[0][action] >= 0 and loss[0][action] <= 1:
            loss[0][action] = 1



        Y = self.learning_rate * loss


        self.model.train_on_batch(X, Y)


    def best_action(self, state):
        "Gets best action based on current state"
        X = state
        Q = self.model.predict(X)
        self.best_Qvalue = np.max(Q)
        return np.argmax(Q)


    def action_probs(self, state):
        X = state
        Q = self.model.predict(X)
        self.best_Qvalue = np.max(Q)
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

    def getIfromRGB(self, rgb):
        "RGB to integer converter"
        red = rgb[:, :, 0]
        green = rgb[:, :, 1]
        blue = rgb[:, :, 2]
        RGBint = (red << 16) + (green << 8) + blue
        return RGBint


    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


    def resize(self, frame):
        frame = imresize(frame, (84, 96))
        # remove 12px from right, because we can go all the way left but not all the way right
        return frame[:, 0:84]


    def _prepro(self, state):
        "Reshape for (1, DIM) as input to Keras"
        state = self.rgb2gray(state)
        state = self.resize(state)

        # show input image
        # plt.imshow(state)
        # plt.gray()
        # plt.show()
        return state.reshape((1,) + state.shape)


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
        state = self._prepro(state)

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

        if self.eps < self.info["Agent"]["eps_min"]:
            self.eps = self.info["Agent"]["eps_min"] # Make sure to maintain a minimum greediness

        if np.random.uniform(0, 1) < self.eps: # Be greedy
            self.action = self.action_space.sample()

        elif info["Agent"]["policy"] == "hardmax":
            self.action = self.Qnetwork.best_action(new_state)

        elif info["Agent"]["policy"] == "softmax":
            probs = self.Qnetwork.action_probs(state)
            if np.sum(probs[0]) < 0.1: # If most options suck
                self.action = self.action_space.sample()
            elif np.sum(probs[0] > 0) < 1: # Make sure to not get an error when all probabilities are 0
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
        self.memory_info = info["Predict_future_n"]
        self.replay_info = info["Replay"]


        self.state_memory = []
        self.state_next_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.cur_size = 0

    def _update_memory(self):
        # if len(self.state_memory) > self.memory_info["size"]:
        if len(self.state_memory) > self.replay_info["memory"]:
            del self.state_memory[0]
            del self.action_memory[0]
            del self.reward_memory[0]

        self.cur_size = len(self.state_memory)

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

    # outdir = '/tmp/random-agent-results'
    # env = wrappers.Monitor(env, directory=outdir, force=True) # this line disables closing
    # env.seed(0)
    return env


# safe deaths and iter
def save_model_params(filename, deaths, iter, avg_q_values, cum_rewards):
    with open("models/" + filename + "_params", 'wb') as fp:
        pickle.dump([deaths, iter, avg_q_values, cum_rewards], fp)



# load default params, or from loaded model if defined
def init_params(info, agent):
    if not info['LoadModel'] or not os.path.isfile("models/" + info['LoadModel'] + "_params"):
        return 0, 0, [], [], agent

    # read in params from loaded model
    with open ("models/" + info['LoadModel'] + "_params", 'rb') as fp:
        [deaths, iter, avg_q_values, cum_rewards] = pickle.load(fp)
        agent.iter = iter
        return deaths, iter, avg_q_values, cum_rewards, agent



class MarioPlotter(object):
    def __init__(self, cum_rewards, avg_qs, deaths, plot_per_x_deaths):
        self.deaths = [0]
        self.cum_rewards = [0]
        self.avg_qs = [0]

        # initialise the plot with loaded params from a previous model, if given
        if deaths >= plot_per_x_deaths:
            for death_n in range(1,deaths+1):
                if death_n % plot_per_x_deaths == 0:
                    self.deaths.append(death_n)
                    # rewards are saved per death, we want to average over plot_per_x_deaths deaths
                    cum_reward = np.sum(cum_rewards[((death_n) - plot_per_x_deaths):(death_n)]) / float(plot_per_x_deaths)
                    cum_Q = np.sum(avg_qs[((death_n) - plot_per_x_deaths):(death_n)]) / float(plot_per_x_deaths)
                    self.avg_qs.append(cum_Q)
                    print ("Calculated cum_reward:", cum_reward)
                    self.cum_rewards.append(cum_reward)

        self.fig = plt.figure()
        self.fig.suptitle("Mario Statistics")

        # Make the cumulative reward plots
        self.ax1 = self.fig.add_subplot(121)
        self.ax1.set_ylabel("Average cumulative reward per " + str(plot_per_x_deaths) + " deaths" )
        self.ax1.set_xlabel("Death #")

        # For live plotting
        self.nowplot, = self.ax1.plot(self.deaths,self.cum_rewards, 'r-')

        # For Q value
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_ylabel("Average best Q value" )
        self.ax2.set_xlabel("Death #")
        self.ax2.yaxis.set_label_position("right")

        # For live plotting
        # self.nowplot2, = self.ax2.plot(range(1,deaths+1),self.avg_qs, 'b-')
        self.nowplot2, = self.ax2.plot(self.deaths, self.avg_qs, 'b-')

        self.fig.canvas.draw()
        plt.show(block=False)


    def __call__(self, cum_reward, avg_q, deaths):
        """Updates the plot"""

        # Set params
        self.deaths.append(deaths)
        self.cum_rewards.append(cum_reward)

        # For cumulative reward
        self.nowplot.set_xdata(self.deaths)
        self.nowplot.set_ydata(self.cum_rewards)

        # Live plot
        self.ax1.relim()
        self.ax1.autoscale_view(True,True,True)

        # Q values
        self.avg_qs.append(avg_q)

        # Update input
        # self.nowplot2.set_xdata(range(1,deaths+1))
        self.nowplot2.set_xdata(self.deaths)
        self.nowplot2.set_ydata(self.avg_qs)

        # Live plot
        self.ax2.relim()
        self.ax2.autoscale_view(True,True,True)

        self.fig.canvas.draw()




# The actual code
N_iters_explore = 1

info = {
    "Game" : 'SuperMarioBros',
    "Worlds" : [1],
    "Levels" : [1], #[1,3,4] level 2 is random shit for all worlds, e.g. water world. See readme
    "Version" : "v1",
    "Plottyplot" : True,
    "Plot_avg_reward_nruns" : 3, # number of runs to average over to show in the plot
    "Network": {"learning_rate": 0.6, "gamma": 0.8},
    "Predict_future_n": {"size" : 7},
    "Replay": {"memory": 100000, "batchsize": 10},
    "Agent": {"type": 1, "eps_min": 0.15, "eps_decay":  2.0*np.log(10.0)/N_iters_explore,
              "policy": "hardmax" #softmax
               },
   "LoadModel" : "SS_test", # False = no loading, filename = loading (e.g. "model_dark_easy_1-5(=worlds)_13(=levels)")
   "SaveModel" : "SS_test" # False= no saving, filename = saving (e.g. "model_dark_easy_1-5(=worlds)_13(=levels)")
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

# save runs for plot
cum_reward = 0   # one reward
cum_rewards = [] # save the cum rewards of all runs
plot_per_x_deaths = info["Plot_avg_reward_nruns"] # to smooth the graph plot the average cum_reward of the last x deaths
avg_q_values = []

# SS params
deaths, iter, avg_q_values, cum_rewards, agent = init_params(info, agent)
Plotter = MarioPlotter(cum_rewards, avg_q_values, deaths, plot_per_x_deaths)


try:
    while True:
        env = init_level(info)
        state = env.reset()

        # run specific vars
        cum_reward = 0
        avg_q = [] # list of avg q values for current state

        while True:
            action = agent.act(state, reward, done)
            state, reward, done, _ = env.step(action)
            reward = calc_reward(reward, done)
            cum_reward += reward
            avg_q += [0] if info["Agent"]["type"] == 0 else [agent.Qnetwork.best_Qvalue] # makes the plot work for both random and deep RL agent
            print("Iter: {} | Reward: {} | Run cumulative reward: {:0.2f} | Deaths: {}".format(iter, reward, cum_reward, deaths))
            iter += 1

            if reward <= -3:
                print ("------DEAD!------")
                deaths += 1

                if deaths % plot_per_x_deaths == 0:
                    # calc avg reward from last plot_per_x_deaths saved deaths
                    avg_reward = np.sum(cum_rewards[((deaths) - plot_per_x_deaths):(deaths)]) / float(plot_per_x_deaths)
                    Plotter(cum_reward, np.mean(avg_q), deaths) # Add death to plot

            # Stops the game
            if done:
                print ("Done")
                cum_rewards.append(cum_reward)
                avg_q_values.append(np.mean(avg_q))
                print ("Avg q values:", avg_q_values)
                env.close()
                break


except KeyboardInterrupt:
    print ("Interrupted by user, shutting down")
except Exception as e:
    traceback.print_exc()
    print ("Unexpected error:", sys.exc_info()[0] , ": ", str(e))
finally:
    # Close the env and write model / result info to disk
    if info['SaveModel'] and info["Agent"]["type"] == 1:
        agent.save_model(info['SaveModel'])
        save_model_params(info['SaveModel'], deaths, iter, avg_q_values, cum_rewards)
    if env:
        env.close()
