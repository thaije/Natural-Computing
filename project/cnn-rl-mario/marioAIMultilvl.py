import gym_super_mario_bros
import numpy as np
from gym import wrappers, logger
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Input, Lambda
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop, Adam
from keras.models import Model
from keras import backend as K
import random, h5py, os.path, pickle, traceback, sys, copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from scipy.misc import imresize
import matplotlib.pyplot as plt


def calc_reward(old_reward, done):
    """Converts default rewards of gym_super_mario_bros package. Calculates new reward not only based on distance"""
    if done and old_reward <= -50: # If done but not dead (or timeout?)
        new_reward = -3
    elif done: # If level finished
        new_reward = 2
    elif old_reward >= 8: # If speed up fast
        new_reward = 1
    elif old_reward > 0: # If move to right
        new_reward = 0.5
    elif old_reward == 0: # If stand still slight punishment
        new_reward = -0.1
    elif old_reward < 0 and old_reward >= -8: # If move to left
        new_reward = -1
    elif old_reward <-8 and old_reward > -50: # If move to left fast
        new_reward = -1.5
    else: # Also dead?
        new_reward = -3

    return new_reward


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state, reward, done):
        return self.action_space.sample()


class Qnetwork:
    def __init__(self, env, info):

        # Training Parameters
        self.training = info["Training"]
        self.plot_q_pred_img = info["Plot_imgs_predicted_q"]
        self.network_info = info["Network"]
        self.replay_info = info["Replay"]
        self.prev_states_count = self.network_info["input_frames"] - 1 # minus current frame

        # Learning parameters
        self.learning_rate = self.network_info["learning_rate"]
        self.gamma = self.network_info["gamma"]

        # Model network function
        self.model = self._build_model(env)
        if info['LoadModel']:
            self.load_model(info['LoadModel'])

        # Extra
        self.best_Qvalue = 0

        self.actions_index = [
            'NOP',
            'Up',
            'Down',
            'Left',
            'Right',
            'Left + A',
            'Left + B',
            'Left + A + B',
            'Right + A',
            'Right + B',
            'Right + A + B',
            'A',
            'B',
            'A + B'
        ]



    def _build_model(self, env):

        input_3D = (4, 84, 96) # nframes x width x height
        frames_input = keras.layers.Input(input_3D, name='frames')

        # normalize input values from 0-255 to 0-1
        normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)

        # Model layout from http://cs229.stanford.edu/proj2016/report/klein-autonomousmariowithdeepreinforcementlearning-report.pdf
        conv1 = Convolution2D(32, (8, 8), strides=(4, 4), padding="same", activation="relu", kernel_initializer="he_uniform", data_format="channels_first")(normalized)
        conv2 = Convolution2D(64, (4, 4), strides=(2, 2), padding="same", activation="relu",kernel_initializer="he_uniform")(conv1)
        conv3 = Convolution2D(128, (3, 3), strides=(1, 1), padding="same", activation="relu",kernel_initializer="he_uniform")(conv2)
        flatty_flat = Flatten()(conv3)

        fully_connected = Dense(512, activation="relu", kernel_initializer="he_uniform")(flatty_flat)
        output_layer = Dense(env.action_space.n, kernel_initializer="he_uniform", activation="linear")(fully_connected)


        # Model from original Deep Q learning paper https://arxiv.org/pdf/1312.5602v1.pdf
        # model = Sequential()
        # model.add(Convolution2D(16, (8, 8), strides=(4, 4), padding="same", activation="relu",kernel_initializer="he_uniform", data_format="channels_last", input_shape=input_3D))
        # model.add(Convolution2D(32, (4, 4), strides=(2, 2), padding="same", activation="relu",kernel_initializer="he_uniform"))
        #
        # model.add(Flatten())
        # model.add(Dense(256, activation="relu", kernel_initializer="he_uniform"))
        # model.add(Dense(env.action_space.n, activation="linear"))

        optim = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        # optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model = Model(input=frames_input, output=output_layer)
        model.compile(loss=huber_loss, optimizer=optim)
        model.summary()

        return model


    def save_model(self, filename):
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


    def _get_reward(self, state_with_prevs, future_states, reward_first):
        """Calculates the reward based on however many states are in the list. So if there are 2 states it looks two in the future, etc.."""

        # TODO: make this work with extra history frames
        # Dims = np.shape(statelist)
        # Qs = self.model.predict(np.squeeze(state_with_prevs))
        # rewards = np.amax(Qs, axis=1)
        # rewards[0] = reward_first
        # rewards = [(self.gamma ** i) * rewards[i] for i in range(Dims[0])]
        #
        # r = np.sum(rewards)
        # return r

        # classic Q learning formula, Q value = reward + reward of next state with discount factor
        pred = np.amax(self.model.predict(state_with_prevs)[0])
        Q_value = reward_first + self.gamma * pred
        # print ("Predicted Q:", pred)

        return Q_value


    def _randbatch(self, History):
        future_look_size = History.fut_n
        n_states_stored = History.cur_size

        # get a random index for an replay item
        idx = np.random.randint(low = self.prev_states_count, high = n_states_stored - future_look_size)

        # retrieve the x previous and n future states
        prev_states = History.state_memory[idx - self.prev_states_count: idx]
        prev_states = np.squeeze(prev_states, axis=0)
        future_states = History.state_memory[idx + 1: (idx + future_look_size)]

        # retrieve the current state + corresponding action and reward
        state = History.state_memory[idx]
        reward = History.reward_memory[idx]
        action = History.action_memory[idx]

        return reward, action, state, prev_states, future_states

    def _plot_prediction(self, Q_target, Qs_predicted, action, state_with_prevs):
        print ("Q target vs Q predicted:", Q_target, Qs_predicted[0][action])
        print ("Shape states:", np.shape(state_with_prevs))
        print ("Action:", action, " = ", self.actions_index[action])

        # plot pictures
        fig = plt.figure(figsize=(20, 8))
        fig.suptitle("Last " + str(self.prev_states_count + 1) + " frames with predicted and target Q val for action")
        for i in range(1, self.prev_states_count + 2):
            img = np.reshape(state_with_prevs[0][i-1], (84,96))
            fig.add_subplot(1, self.prev_states_count + 1, i)
            plt.imshow(img)
            plt.gray()
        plt.show()


    def replay(self, History):
        """ Train model on some random replayed experiences """

        # Creates n batches of size frames + n_future_look and fits model
        # The main part that makes this slow is the actual fitting! (not the loopy loops)
        n_replay = self.replay_info["batchsize"]
        if History.cur_size > History.fut_n + self.prev_states_count + self.network_info["warmup"]:
            X= []
            Y =[]
            for i in range(n_replay):
                reward, action, state_0, prev_states, future_states = self._randbatch(History)

                # 1 input item = 4 frames (given as channels) of 84 x 96 px
                state_with_prevs = np.append(prev_states, state_0, axis=0)
                state_with_prevs = np.reshape(state_with_prevs, (1, self.prev_states_count + 1, 84, 96))

                # get the actual reward of the current state + future rewards
                Q_target = self._get_reward(state_with_prevs, future_states, reward)

                # predict the Q value of the current frame with last x previous frames
                Qs_predicted = self.model.predict(state_with_prevs)

                # if i == 1:
                    # print ("Training. Action: {} {} | Reward: {} | Q target: {:0.2f} | Q predicted: {:0.2f}".format(action, self.actions_index[action], reward, Q_target, Qs_predicted[0][action]))

                # plot the x history frames with predicted Q, target Q, and action, to see what prediction is like
                if self.plot_q_pred_img:
                    self._plot_prediction(Q_target, Qs_predicted, action, state_with_prevs)

                # We feed the network the correct Q value for the state-action pair
                # we know / calculated. The other state-action Q values are kept the
                # same as Q_predicted so they don't influence the network
                Qs_predicted[0][action] = Q_target # + ?

                # The gradient descent step with the loss is calculated and
                # done automatically by Keras
                X.append(state_with_prevs)
                Y.append(Qs_predicted)

            X = np.squeeze(X)
            Y = np.squeeze(Y)

            self.model.fit(X, Y, batch_size = self.replay_info["batchsize"], verbose=0, epochs=1)



    def best_action(self, state, history):
        """ Gets best action based on current state"""

        # Wait untill we have atleast x frames in the replay memory before training the network (warmup) if we are in training mode
        # We also need atleast x history frames to get a prediction from the CNN
        if (history.cur_size < self.prev_states_count) or (history.cur_size < self.network_info["warmup"] and self.training):
            return -1

        # concatenate the last frame with x history frames
        prev_states = history.state_memory[-self.prev_states_count:]
        prev_states = np.squeeze(prev_states, axis=0)
        state_with_prevs = np.append(prev_states, state, axis=0)
        X = np.reshape(state_with_prevs, (1, self.prev_states_count + 1, 84, 96))

        Q = self.model.predict(X)
        self.best_Qvalue = np.max(Q)
        return np.argmax(Q)


    def action_probs(self, state):
        X = state
        Q = self.model.predict(X)
        self.best_Qvalue = np.max(Q)
        Q[Q <0] = 0

        return (Q/np.sum(Q))


# From https://becominghuman.ai/beat-atari-with-deep-reinforcement-learning-part-2-dqn-improvements-d3563f665a2c
def huber_loss(a, b, in_keras=True):
    """ Computes the Huber Loss: MSE for low values and MAE for large values """
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term


class Qagent(object):
    """Qagent"""
    def __init__(self, env, info):
        self.info = info
        self.action_space = env.action_space
        self.Qnetwork = Qnetwork(env, info)

        self.start = True
        self.iter = 0
        self.state_hist = []
        self.action = self.action_space.sample()
        self.eps_decay = self.info["Agent"]["eps_decay"]
        self.eps = self.info["Agent"]["eps_start"]

        self.History = Memory(info) # replay memory


    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


    def resize(self, frame):
        frame = imresize(frame, (84, 96)) # original dim (224, 256)
        # Make the frame square. Remove 12px from left, because we
        # want to see as far ahead as possible while speedrunning
        # frame = frame[:, 12:96]
        return frame


    def _prepro(self, state):
        """Reshape for (1, DIM) as input to Keras"""
        state = self.rgb2gray(state)
        state = self.resize(state)

        # show input image
        # plt.imshow(state)
        # plt.gray()
        # plt.show()
        return state.reshape((1,) + state.shape)


    def _update_statelist(self, state):
        """To store previous state and new state. Previous state is associated with the current reward"""
        # if we are just starting, just append the current state as history
        if self.start:
            self.state_hist.append(state)
            self.state_hist.append(state)
        else:
            del self.state_hist[0]
            self.state_hist.append(state)


    def act(self, state, reward, done):
        """ Train Q network on replayed experiences, and predict best action for current state"""

        # preprocess the state and save it to history
        state = self._prepro(state)
        self._update_statelist(state)

        # If first iter then there's no history to learn from
        if self.start:
            self.action = self.action_space.sample()
            self.iter += 1
            self.start = False
            return self.action

        # save current state in the replay memory
        old_state = self.state_hist[0]
        new_state = self.state_hist[1]
        self.History.append_to_memory(old_state, new_state, self.action, reward)

        ### Train Q network on replay memory items
        if self.info["Training"]:
            self.Qnetwork.replay(self.History)

        # Update Greedy epsilon
        if self.iter > self.info["Network"]["warmup"]:
            self.eps = self.info["Agent"]["eps_start"] - (self.eps_decay * (self.iter - self.info["Network"]["warmup"]) )

            # Make sure to maintain a minimum greediness
            if self.eps < self.info["Agent"]["eps_min"]:
                self.eps = self.info["Agent"]["eps_min"]


        ### Generate a new action for the current state
        # Be greedy
        if np.random.uniform(0, 1) < self.eps:
            self.action = self.action_space.sample()

        elif info["Agent"]["policy"] == "hardmax":
            # returns either -1 (not enough frames yet), or the best action to do
            self.action = self.Qnetwork.best_action(new_state, self.History)
            if self.action == -1:
                self.action = self.action_space.sample()

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


# The replay memory
class Memory:
    def __init__(self, info):
        # Memory info
        self.fut_n = info["Network"]["predict_future_n"]
        self.replay_info = info["Replay"]

        self.state_memory = []
        self.state_next_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.cur_size = 0

    def _update_memory(self):
        # Delete the oldest item if the replay memory becomes too big
        if len(self.state_memory) > self.replay_info["memory"]:
            del self.state_memory[0]
            del self.action_memory[0]
            del self.reward_memory[0]

        self.cur_size = len(self.state_memory)

    def append_to_memory(self, state, state_next, action, reward):
        # save previous state, action and reward
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

        # save current state
        self.state_next_memory = state_next
        self._update_memory()



# load a random world / level and return the env
def init_level(info):
    world = random.choice(info['Worlds'])
    lvl = random.choice(info['Levels'])

    env = gym_super_mario_bros.make(info['Game'] + "-" + str(world) + "-" + str(lvl) + "-" + info['Version'])

    return env


# safe current param values of this model to a pickle file
def save_model_params(filename, deaths, iter, avg_q_values, cum_rewards, lvl_completed_counter):
    with open("models/" + filename + "_params", 'wb') as fp:
        pickle.dump([deaths, iter, avg_q_values, cum_rewards, lvl_completed_counter], fp)



# load default params, or from model file if defined
def init_params(info, agent):
    if not info['LoadModel'] or not os.path.isfile("models/" + info['LoadModel'] + "_params"):
        return 0, 0, [], [], agent, 0

    # read in params from loaded model
    with open ("models/" + info['LoadModel'] + "_params", 'rb') as fp:
        [deaths, iter, avg_q_values, cum_rewards, lvl_completed_counter] = pickle.load(fp)
        agent.iter = iter
        return deaths, iter, avg_q_values, cum_rewards, agent, lvl_completed_counter


def init_env(info):
    env = init_level(info)
    random_levels = True
    if len(info["Worlds"]) == 1 and len(info["Levels"]) == 1:
        random_levels = False

    if info["Network"]["warmup"] < 0 or info["Network"]["input_frames"] < 1 or info["Network"]["predict_future_n"] < 1:
        print ("Error, invalid parameter value")
        sys.exit(1)

    if not info["Training"] and not info['LoadModel']:
        print ("Error - Can't have training mode False with no trained model selected")
        sys.exit(1)

    if info["Agent"]["type"] == 0:
        agent = RandomAgent(env.action_space)
    else:
        agent = Qagent(env, info)

    return env, random_levels, agent


# Live plotter for rewards-deaths and best_Q-deaths, plotted every x deaths as avg of every x deaths
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
        self.nowplot2.set_xdata(self.deaths)
        self.nowplot2.set_ydata(self.avg_qs)

        # Live plot
        self.ax2.relim()
        self.ax2.autoscale_view(True,True,True)
        self.fig.canvas.draw()




# Google settings:
# N_iters_explore = 1 000 000
# eps_min: 0.1
# Replay memory = 1 000 000
# Total frames trained:  10 000 000
# last 4 frames as input to same DNN

# Our settings:
# N_iters_explore = 500 000
# eps_min: 0.15
# memory: 10000
# total frames trained: 500k?

# Settings https://github.com/aleju/mario-ai
# N_iters_explore = 400 000
# eps_min: 0.1
# memory: 250 000
# total frames trained: 500k?
# last 4 frames to seperate layer. Last framer higher resolution to other layer. Other layer for last 4 actions. Merge


# The actual code
N_iters_explore = 325000

info = {
    "Game" : 'SuperMarioBros',
    "Worlds" : [1], # 1=buizen, 5=enemies, 6=gaten
    "Levels" : [1], #[1,3,4] level 2 is random shit for all worlds, e.g. water world. See readme
    "Version" : "v2",
    "Plottyplot" : True, # plot rewards/deaths and best_q/deaths
    "Training" : True, # Training or demo mode. False will load a trained DQN model defined below, and execute without further training. True will train DQN model
    "Plot_avg_reward_nruns" : 3, # number of runs to average over to show in the plot
    "Plot_imgs_predicted_q": False, # Will plot input_frames + predicted/target Q and action after warmup

    # Gamma = discount_rate. Input_frames = current frame + x history frames. Warmup = don't train on replay exp untill x items are in the replay mem
    # Predict_future_n =  Look n states into future to calc Q_val. n=1 = normal Q_val calculation
    "Network": {"learning_rate": 0.99, "gamma": 0.9, "input_frames": 4, "warmup": 25000, "predict_future_n": 1},
    "Replay": {"memory": 250000, "batchsize": 32}, # train on {batchsize} replay experiences per iteration
    "Agent": {"type": 1, "eps_start": 1.0, "eps_min": 0.1, "eps_decay": (1.0-0.15)/N_iters_explore,
              "policy": "hardmax"
               },
   "LoadModel" : "t_100000_params", # False = no loading, filename = loading (e.g. "test_model")
   "SaveModel" : "t_v2", # False = no saving, filename = saving (e.g. "test_model")
}



# load mario lvl and init agent
env, mult_lvls, agent = init_env(info)
mult_lvls = True

reward = 0
done = False

# save statistics for plot
cum_reward = 0   # one reward
cum_rewards = [] # save the cum rewards of all runs
plot_per_x_deaths = info["Plot_avg_reward_nruns"] # to smooth the graph plot the average cum_reward of the last x deaths
avg_q_values = []

deaths, iter, avg_q_values, cum_rewards, agent, lvl_completed_counter = init_params(info, agent)
Plotter = MarioPlotter(cum_rewards, avg_q_values, deaths, plot_per_x_deaths)


try:
    while True:
        if mult_lvls:
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

            # makes the plot work for both random and deep RL agent
            if info["Agent"]["type"] == 0 or iter < info["Network"]["warmup"]:
                avg_q += [0]
            else:
                avg_q += [agent.Qnetwork.best_Qvalue]
            print("Iter: {} | Reward: {} | Run cumulative reward: {:0.2f} | Deaths: {} | Lvl completions: {} | Eps: {:0.2f}".format(iter, reward, cum_reward, deaths, lvl_completed_counter, agent.eps))
            iter += 1

            if reward <= -3:
                print ("------DEAD!------")
                deaths += 1

                if deaths % plot_per_x_deaths == 0:
                    # calc avg reward from last plot_per_x_deaths saved deaths
                    avg_reward = np.sum(cum_rewards[((deaths) - plot_per_x_deaths):(deaths)]) / float(plot_per_x_deaths)
                    Plotter(cum_reward, np.mean(avg_q), deaths) # Add death to plot

            # autosave model every 100k iterations
            if info['SaveModel'] and iter % 100000 == 0 and info["Training"]:
                agent.save_model(info['SaveModel'] + "_" + str(iter))
                save_model_params(info['SaveModel'] + "_" + str(iter), deaths, iter, avg_q_values, cum_rewards, lvl_completed_counter)

            # Stops the game
            if done:
                cum_rewards.append(cum_reward)
                avg_q_values.append(np.mean(avg_q))
                if mult_lvls:
                    env.close()
                if reward > -3:
                    lvl_completed_counter += 1
                print ("Done.")
                break


except KeyboardInterrupt:
    print ("Interrupted by user, shutting down")
except Exception as e:
    traceback.print_exc()
    print ("Unexpected error:", sys.exc_info()[0] , ": ", str(e))
finally:
    # Close the env and write model / result info to disk
    if info['SaveModel'] and info["Agent"]["type"] == 1 and info["Training"]:
        agent.save_model(info['SaveModel'])
        save_model_params(info['SaveModel'], deaths, iter, avg_q_values, cum_rewards, lvl_completed_counter)
    if env:
        env.close()
