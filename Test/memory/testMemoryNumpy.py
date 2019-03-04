

from Agents.Core.ReplayMemory import ReplayMemory, Transition
#from ..Agents.Core.ReplayMemory import ReplayMemory, Transition
import torch
import numpy as np
import pickle

state1 = np.random.rand(5, 5)
state2 = np.random.rand(5, 5)
state3 = np.random.rand(5, 5)
state4 = np.random.rand(5, 5)



tran1 = Transition(state1, 1, state2, 1)
tran2 = Transition(state3, 2, state4, 2)
memory = ReplayMemory(10)
memory.push(tran1)
memory.push(tran2)
print(memory)

file = open('memory.pickle','wb')
pickle.dump(memory, file)
file.close()

with open('memory.pickle','rb') as file:
    memory2 = pickle.load(file)

print(memory2)


