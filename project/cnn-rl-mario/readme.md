

# run code

- `source ~/tensorflow/bin/activate`

# links
https://github.com/Kautenja/gym-super-mario-bros#individual-levels

# Todo:
- plotting
    + reward by trial (average of 5)
    + average Q value section 5 https://arxiv.org/pdf/1312.5602v1.pdf
    + load params of previous run and include in plot
    - run for 5 million iterations or so
    - fix bg fceux
- reward =
    - paper -> distance to the right + total game score. Primary = get to finish. Second=get high score
    + separate reward for get live, finish level, loose live
    + reward distance^2 * lvl, lvl complete bonus, death
    + fast moving right / left. Low negative reward for dying? https://github.com/aleju/mario-ai
    - reward finished level / time out same?
- actions:
    - change env to minimize action space https://github.com/Naereen/gym-nes-mario-bros/blob/master/src/nesgym/mario_bros_env.py
- run multiple games in parallel
    - multithreading?
    - https://github.com/Naereen/gym-nes-mario-bros/blob/master/src/run-mario.py
- network adaptations:
    - exploration vs exploitation
    - save / load replay memory?
    - (Double) deep Q-learning https://github.com/Naereen/gym-nes-mario-bros/tree/master/src/dqn / http://cs229.stanford.edu/proj2016/report/klein-autonomousmariowithdeepreinforcementlearning-report.pdf
    + Ours 128 -> 64 > 32 > 16. Theirs 32 > 64 > 64 ? Check different architectures.
    + preprocessing to grey scale etc. preprocessing see 4.1 https://arxiv.org/pdf/1312.5602v1.pdf / http://cs229.stanford.edu/proj2016/report/klein-autonomousmariowithdeepreinforcementlearning-report.pdf

# notes
- trainen op lvl 1/3, testen lvl 2
- gets stuck on bonus lvl, q value drop down
- add ingame score for motivation to kill enemies / learn what enemies are
- we use frame skipping
- enemies, main character, surface can have same colour
- start and pavement same colour
- Use spatial transformer to transpose image to self / mario? https://github.com/aleju/mario-ai

algorithm:
Input = grey 84x84 area (does not need to be square), stack of last 4 frames + last 4 actions?


Recent experiences are saved in a replay memory. An experience = Dataset D of pooled experiences e_t = (state_t, action_t, reward_t, state_t+1).
Q learning updates or done on random experiences from this memory, to remove correlation between frames (go left, only train on go left. Instead smooth actions).
Normally for Q learning, an action-value function is estimated for every sequence. But now many sequences possible -> create approximate estimator to generalize over sequences -> neural network
Used RMSprop gradient descent with minibatches of 32 / or MSE?


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
- maybe add a reward for passing a place where it died many times?
