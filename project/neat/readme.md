
# install requirements
- Keras
- gym
- https://github.com/Kautenja/gym-super-mario-bros
- Python 3.x
- numpy, scipy


# run code
- To run the best model `python3 best_run.py genome_good.pkl` can be executed
- run `python3 mario.py` starts a new training run that trains until the user stops it, the training progress will be logged. Models are saved when the maximum fitness has improved. Saved models can be continued to be trained by adding the filename when calling mario.py
- parse_log.py can be run after training to visualize the training progress.


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




# links
- https://github.com/Kautenja/gym-super-mario-bros#individual-levels\
- https://gist.github.com/d12frosted/7471e2123f10485d96bb
