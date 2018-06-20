
# install requirements
- Keras
- gym
- https://github.com/Kautenja/gym-super-mario-bros
- Python 3.x
- numpy, scipy


# run code
- By default the program will run a pre-trained model. See the next section to change functionality.
- run `python3 marioAIMultilvl.py`


# Settings
There are a number of parameters in `marioAIMultilvl.py` which can change the functionality, located in the `info` variable around line 620:
- Train a new model: set Training to True, LoadModel to False, and SaveModel to a string with the desired filename.
- World, Levels and Version correspond to the Mario worlds, levels and versions as specified in the original Super Mario Bros Gym package: https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/smb_env.py
- Run a model without training: set Training to False, LoadModel to the desired model.
- See the Network, Agent and Replay parameters for tweaking variable values of these.



# Levels
- `SuperMarioBros-<world>-<level>-v<version>`
- `<world>` is a number in {1, 2, 3, 4, 5, 6, 7, 8}
- `<level>` is a number in {1, 2, 3, 4} indicating the level within a world
- World/level combos:
    - World [1,2,3,4,6,7,8], level 1: outside lvl in nighttime, brown brick, black bg
    - World [1,4], level 2: underground lvl, blue bricks / enemies, black bg
    - World [2,7], level 2: swim lvl, blue bg, different physics
    - world [3,5,6,8], level 2: outside lvl in nighttime, brown brick, black bg, new enemies
    - World [1,2,3,4,5,7,8], level 3: outside lvl in nighttime, brown brick, green/yellow paddos, black bg
    - world 6, level 3: outside lvl in nighttime, white brick, black bg
    - World [1,2,3,4,5,6,7,8], level 4: underground lvl, white bricks, lava, black bg

    Things to note:
    - World 4: new enemies: eating plants, chicken turtles
    - World 5: new enemies: cannons
    - World 8: new enemies: black walking monsters, flying/jumping chicken turtles, mofuckers in clouds dropping hedgehogs
- if you create multiple gym_super_mario_bros environments, the world/level of the last one overwrites the first one (world/level).
So you can't have two different worlds/levels running in the emulator from the same script at the same time
- if you run multiple emulators from one script at once, the script will run 10(?) frames in the first, than 10(?) in the second, back-and-forth.


# Todo:
- run for 5 million iterations or so
- reward: add ingame score for motivation to kill enemies / learn what enemies are
- actions: change env to minimize action space https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/smb_env.py
- network adaptations:
    - save / load replay memory?
    - (Double) deep Q-learning https://github.com/Naereen/gym-nes-mario-bros/tree/master/src/dqn / http://cs229.stanford.edu/proj2016/report/klein-autonomousmariowithdeepreinforcementlearning-report.pdf
    - Test different architectures.
- test by training on multiple levels
- maybe add a reward for passing a place where it died many times?
- steven reward func
- q value mask?
- implement separate network for generating target Q
- multiple distance based epsilons

# notes
- train on lvl 1/3, test on lvl 2
- gets stuck on bonus lvl, q value drops down
- we use frame skipping
- enemies, main character, surface can have same colour
- start and pavement same colour
- Use spatial transformer to transpose image to self / mario? https://github.com/aleju/mario-ai
- some game elements only encountered later on (holes in the ground, etc.)
- Colours:
    - Mario not very distinctive -> give other colour?
    - Enemies same colour as surface
    - As such -> better in swim lvl? enemies / mario / surface etc distinct colour
- architecture
- 14 actions



# DQN overview
- Get action for current state:
    - Give 4 frames as input (current + 3 history)
    - Also give last x actions? no
    - Get list of Q values for each action
    - Use best one for next state
    - Save experience in replay memory

- Training on x experiences from replay memory:
    - Get Q value of the current state:
        - reward + gamma (discount) * predict_next_state (^x?)

- Predict Q Value
    - returns list of Q values (expected reward) per action

- Input = frame + 3 frames history
- Target = predicted Q_values + corrected Q(s,a) Q_value
- Difference = (huber)loss (automatically calculated in model)


history = for spotting movement
Training on replay memory = to remove correlation between frames (go left, only train on go left. Instead smooth actions).

Normally for Q learning, an action-value function is estimated for every sequence. But now many sequences possible -> create approximate estimator to generalize over sequences -> neural network

Difference:
Prediction is known for 1 state/action/reward/state combo. The “correct” Q value of other actions in that state are not known, so keep them the same in the prediction to have them not affect the model. As such, calc the loss between Q_predicted and Q_predicted with corrected reward for the known action.

Only use last 4 frames as input:
The main drawback of this type of architecture is that a separate forward pass is required to compute the Q-value of each action, resulting in a cost that scales linearly with the number of actions. We instead use an architecture in which there is a separate output unit for each possible action, and only the state representation is an input to the neural network. The outputs correspond to the predicted Q-values of the individual action for the input state.

# links
- https://github.com/Kautenja/gym-super-mario-bros#individual-levels
- https://github.com/Naereen/gym-nes-mario-bros/tree/master/src/dqn
- http://cs229.stanford.edu/proj2016/report/klein-autonomousmariowithdeepreinforcementlearning-report.pdf
- https://github.com/aleju/mario-ai
- https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
