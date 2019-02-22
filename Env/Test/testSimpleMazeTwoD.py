from Env.CustomEnv.SimpleMazeTwoD import SimpleMazeTwoD
import random
import matplotlib.pyplot as plt
import numpy as np
env = SimpleMazeTwoD('map1D.txt')
fig = plt.figure(1)
ax = plt.axes()
env.plot_map(ax)
plt.show()
state = env.reset()


print(state)
stateSet = []
stateSet.append(state)
for i in range(1000):
    action = random.randint(0, env.nbActions - 1)
    state, reward, done, _ = env.step(action)
    #print('step ' + str(i))
    #print(state)
    #print(reward)
    #print(done)
    stateSet.append(state)
    if done:
        break


print(stateSet)
fig = plt.figure(2)
ax = plt.axes()
env.render_traj(stateSet, ax)
plt.show()
