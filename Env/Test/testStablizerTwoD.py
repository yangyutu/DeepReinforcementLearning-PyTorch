from Env.CustomEnv.StablizerTwoD import StablizerTwoD
import random
import matplotlib.pyplot as plt

env = StablizerTwoD()
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
