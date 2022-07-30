###

This folder contains training and evaluation scripts we used to generate results in the paper "Hierarchical planning with deep reinforcement learning for  three-dimensional navigation of microrobots in blood vessels".
A comprehensive description on the training procedure is described in the supplementrary information coming with the paper. 


### Run the training script
You can run the training script in NavigationExamples/Multimaps/DDPGHER_3DCNN.py to train the lower-level deep reinforcement learning controller. The training script takes config.json as the running configuration. 

This training script utilizes a Actor-Critic network structure to learn optimal policy to interact with the environment. The networks utilize 3D CNN to learn state representations. 

### Run the evaluation script

The trained reinforcement learning controller can be evaluated together with a high-level planner using script in NavigationExamples/Multimaps/TrainTwoObsPenalNew/testSet under different settings.


### Simulator
The simulated is based on C++ and python binding. The computation heavy routines are taken care by C++ functions residing in (ActiveParticle3DSimulator.h, ActiveParticle3DSimulator.cpp) and the binding file (ActiveParticle3DSimulatorPython.cpp). We also provided a Makefile to compile them into a python package. 

This simulation package is called and used in activeParticle3DEnv.py, which is a python based environment for reinforcement learning training purpose. Specifically, this environement implement function step(action), which simulates the environment one step forward given the action of the agent at the current step.








