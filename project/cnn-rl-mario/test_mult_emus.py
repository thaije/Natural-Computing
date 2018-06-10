import gym_super_mario_bros as gym
import gym_super_mario_bros as gym2

from time import sleep


env = gym.make('SuperMarioBros-1-1-v1') # should be different from second world
env2 = gym2.make('SuperMarioBros-1-2-v1') # this one overwrites first
state = env.reset()
state = env2.reset()
env.close()

done = True
done2 = True
for step in range(5000):


    if done:
        state = env.reset()
    if done2:
        state2 = env2.reset()

    # should both be updated at same time, but emulators are never updated at the same time
    state, reward, done, info = env.step(env.action_space.sample())
    state2, reward2, done2, info2 = env2.step(env2.action_space.sample())

env.close()
env2.close()
