from Env.CustomEnv.StablizerOneD import StablizerOneD
import random
import matplotlib.pyplot as plt
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

numWorkers = 6
config = {}
envs = []
for i in range(numWorkers):
    env = StablizerOneD()
    envs.append(env)

nbActions = envs[0].nbActions

# we need a wrapper
def make_env(config, i):
    def _thunk():
        env = StablizerOneD(config, i)

        return env
    return _thunk


envs = SubprocVecEnv([make_env(config, i) for i in range(numWorkers) ])

state = envs.reset()
stateSet = []
doneList = []
rewardList = []
stateSet.append(state)
for i in range(1000):
    actionList = []
    for _ in range(numWorkers):
        action = random.randint(0, nbActions - 1)
        actionList.append(action)

    # note that in vector envs: if one env finishes, it will automatically reset and start a new episode
    state, reward, done, _ = envs.step(actionList)
    stateSet.append(state)
    doneList.append(done)
    rewardList.append(reward)
#    if done:
#        break


print(stateSet)

