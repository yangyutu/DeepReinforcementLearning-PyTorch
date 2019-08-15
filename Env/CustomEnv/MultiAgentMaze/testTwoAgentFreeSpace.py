from Env.CustomEnv.MultiAgentMaze.TwoAgentCooperativeTransport import TwoAgentCooperativeTransport
import random

# first construct the neutral network
config = dict()

config['trainStep'] = 5000
config['epsThreshold'] = 0.5
config['epsilon_start'] = 0.5
config['epsilon_final'] = 0.05
config['epsilon_decay'] = 500
config['episodeLength'] = 200
config['numStages'] = 6
config['targetNetUpdateStep'] = 10
config['memoryCapacity'] = 10000
config['trainBatchSize'] = 64
config['gamma'] = 0.99
config['learningRate'] = 0.0001
config['netGradClip'] = 1
config['logFlag'] = False
config['logFileName'] = 'SimpleMazeLog/traj'
config['logFrequency'] = 500
config['priorityMemoryOption'] = False
config['netUpdateOption'] = 'doubleQ'
config['netUpdateFrequency'] = 1
config['priorityMemory_absErrUpper'] = 5
config['device'] = 'cpu'
config['mapWidth'] = 6
config['mapHeight'] = 6

env = TwoAgentCooperativeTransport(config)

nSteps = 100
numAgents = 2

N_S = env.stateDim
N_A = env.nbActions

state = env.reset()
print(state)

for i in range(nSteps):
    print(i)
    actions = []
    for n in range(numAgents):
        actions.append(random.randint(0, 3))
    print(actions)
    state, reward, done, info = env.step(actions)
    print(state)