

import gym
import random


# https://gym.openai.com/docs/
# https://gym.openai.com/envs/CartPole-v0/
# https://github.com/openai/gym/wiki/CartPole-v0

# Num	Observation	Min	Max
# 0	Cart Position	-2.4	2.4
# 1	Cart Velocity	-Inf	Inf
# 2	Pole Angle	~ -41.8°	~ 41.8°
# 3	Pole Velocity At Tip	-Inf	Inf


# Num	Action
# 0	Push cart to the left
# 1	Push cart to the right

# Reward is 1 for every step taken, including the termination step

env = gym.make('CartPole-v0')

env = env.unwrapped

env.reset()
print(env.state)
print(env.action_space)
# Discrete(2), action can take 0 or 1. means left or right
print(env.observation_space)
# observation will be an array of 4 numbers
print(env.observation_space.high)
print(env.observation_space.low)

for epi in range(10):
    env.reset()
    for _ in range(1000):
        env.render()
        action = random.randint(0, 1)
        observation, reward, done, info = env.step(action) # take a random action
        print(observation)
        if done:
            env.close()
            break
