from gym.envs.classic_control.cartpole import CartPoleEnv


class CartPoleEnvCustom(CartPoleEnv):
    def __init__(self):
        super(CartPoleEnvCustom, self).__init__()
        self.stepCount = 0
        self.endStep = 200

    def step(self, action):
        self.stepCount += 1
        state, reward, done, _ = super(CartPoleEnvCustom, self).step(action)

        if done:
            reward = -1
        if not done and self.stepCount > self.endStep:
            done = True
        return state, reward, done, {}

    def reset(self):
        state = super(CartPoleEnvCustom, self).reset()
        self.stepCount = 0
        return state