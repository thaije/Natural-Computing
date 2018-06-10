

# run code

- `source ~/tensorflow/bin/activate`

# links
https://github.com/Kautenja/gym-super-mario-bros#individual-levels

# Todo:
- save trained model
- play trained model
+ run game on multiple kinds of levels
- run multiple games in parallel?


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
