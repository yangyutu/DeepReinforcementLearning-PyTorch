

import gym



# https://gym.openai.com/docs/
# https://gym.openai.com/envs/MountainCar-v0/
# https://github.com/openai/gym/wiki/MountainCar-v0

# Num	Observation	Min	Max
# 0	position	-1.2	0.6
# 1	velocity	-0.07	0.07


# Num	Action
# 0	push left
# 1	no push
# 2	push right

# -1 for each time step, until the goal position of 0.5 is reached.
# As with MountainCarContinuous v0, there is no penalty for climbing the left hill,
# which upon reached acts as a wall.

env = gym.make('MountainCar-v0')

env = env.unwrapped

env.reset()
print(env.action_space)
# Discrete(2), action can take 0 or 1. means left or right
print(env.observation_space)
# observation will be an array of 4 numbers
print(env.observation_space.high)
print(env.observation_space.low)


for _ in range(1000):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print(observation)
    if done:
        break
