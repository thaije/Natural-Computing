import gym_super_mario_bros
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v1')


done = True
for step in range(5000):

    if done:
        state = env.reset()

    # should both be updated at same time, but emulators are never updated at the same time
    state, reward, done, info = env.step(env.action_space.sample())

env.close()
