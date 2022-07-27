# DeepReinforcementLearning-PyTorch
PyTorch implementation of classical deep reinforcement learning algorithms

## Implemented Core Algorithms:
* Vanilla Deep Q-Learning (DQN)
  - Human Level Control Through Deep Reinforement Learning [[Publication]](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/)
* Double Deep Q-Learning (Double DQN)
  - Deep Reinforcement Learning with Double Q-learning [[Publication]](https://arxiv.org/abs/1509.06461)
* Dueling Deep Q-Learning (Dueling DQN)
  - Dueling Network Architectures for Deep Reinforcement Learning [[Publication]](https://arxiv.org/abs/1511.06581)
* Hindsight Experience Replay
  - Hindsight Experience Replay [[Publication]](https://papers.nips.cc/paper/7090-hindsight-experience-replay.pdf)
  
* Prioritized Experience Replay [[Publication]](https://arxiv.org/abs/1511.05952?context=cs)
* Synchronous Deep Q-Learning (SDQN)
* REINFORCE
* Deep Deterministic Policy Gradient (DDPG)
  - Continuous control with deep reinforcement learning [[Publication]](https://arxiv.org/pdf/1509.02971.pdf)
* Asynchronous/Synchronous Advantage Actor Critic (A3C, A2C) [[Publication]](https://arxiv.org/pdf/1602.01783.pdf)
* TD3 [[Publication]](https://arxiv.org/pdf/1802.09477.pdf)
* Soft Actor Critic (SAD) [[Publication]](https://arxiv.org/pdf/1801.01290.pdf)
* Stacked DQN/DDPG/SAC


## Enhancements:
* Hindsight Experience Replay [[Publication]](https://papers.nips.cc/paper/7090-hindsight-experience-replay.pdf)
* Prioritized Experience Replay [[Publication]](https://arxiv.org/abs/1511.05952?context=cs)
* Noisy Networks for Exploration [[Publication]](https://arxiv.org/abs/1706.10295)


## Application examples:

* Efficient Navigation of Active Particles in an Unseen Environment via Deep Reinforcement Learning [[Publication]](https://arxiv.org/abs/1906.10844)
* Hierarchical planning with deep reinforcement learning for three-dimensional navigation of microrobots in blood vessels (under review)

## Custom envs

* 1D stablizer, 2D stabilizer, and multi-Dim stabilizer
* maze with static obstacles and stchastic/deterministic agent
* maze with dynamic obstacles and stchastic/deterministic agent
* finanical portfolio engineering env (for hedging and investment)
* colloidal assembly env
* 3D blood vessel navigation environment


## Third party envs
* [[pybullet]](https://pybullet.org/wordpress/)

## Cited as
@misc{Yang2019,
  author = {Yuguang Yang},
  title = {DRL-Pytorch},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yangyutu/DeepReinforcementLearning-PyTorch}}
}
