import gym_super_mario_bros
from matplotlib import pyplot as plt
import numpy as np
import sys
import mario
from mario import *
import pickle as pkl

if __name__ == '__main__':

	if len(sys.argv) <= 1:
		print('Needs Filename')
		raise RuntimeWarning
	filename = sys.argv[1]

	inp_size = [224, 256, 1]

	env = gym_super_mario_bros.make('SuperMarioBros-v2')
	if 'pool' in filename:
		pool = mario.load_pool(filename)
		genome = pool.get_best_genome()
	else:
		genome = mario.load_pool(filename)
	print(genome.fitness)

	fitnesses = []
	speeds = []
	pos_info = {'pos':[], 'curr': 0, 'best': 0}
	for i in range(4):
		pos_since_dead = []
		current_frame = 0
		state = env.reset()
		mario.initialize_genome(genome, inp_size)
		state, reward, done, info = env.step(env.action_space.sample())
		while not done:

			if not (current_frame%5):
				state = state/255
				output = evaluate_genome(genome,imresize(mario.rgb2gray(state),inp_size).flatten())
				#action=env.action_space.sample()
			probs = output/np.sum(output)
			if np.any(np.isnan(probs)):
				probs = None	

			action = np.random.choice(range(len(output)),p=probs)
			state, reward, done, info = env.step(action)
			current_frame += 1

			if reward <= -50:
				if len(pos_since_dead) > 1:
					speeds.append(np.sum(pos_since_dead)/current_frame)
					print('Speed: '  + str(speeds[-1]))
					fitnesses.append(np.sum(pos_since_dead))
					print('Fitness ' + str(fitnesses[-1]))
				pos_since_dead = []
			else:
				pos_since_dead.append(reward)
				mario.update_pos(reward, pos_info)

			if done:
				state = env.reset()
				
				pos_info = {'pos':[], 'curr': 0, 'best': 0}

					   
			
	fitnesses = fitnesses[:-2]
	speeds = speeds[:-2]
	env.close()
	pkl.dump(fitnesses,open('fitnesses_random.pkl', 'wb'))
	pkl.dump(speeds,open('speeds_random.pkl', 'wb'))
	print(np.mean(fitnesses))
	print(np.mean(speeds))