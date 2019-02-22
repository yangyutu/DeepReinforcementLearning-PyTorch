

from Agents.Core.ReplayMemory import ReplayMemory, Transition
#from ..Agents.Core.ReplayMemory import ReplayMemory, Transition
import torch

tran1 = Transition(1, 1, 1, 1)
tran2 = Transition(2, 2, 2, 2)
memory = ReplayMemory(10)
memory.push(tran1)
memory.push(tran2)
memory.push(3,3,3,3)
print(memory)

memory.write_to_text('memoryOut.txt')

toTensor = memory.totensor()
toTensor2 = torch.tensor(memory.sample(2))
for i in range(5, 50):
    tran = Transition(i,i,i,i)
    memory.push(tran)

print(memory)
memory.clear()
print(memory)
print(toTensor)
print(toTensor2)
