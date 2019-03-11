from gym.envs.classic_control.mountain_car import MountainCarEnv


class MountainCarEnvCustom(MountainCarEnv):
    def __init__(self):
        super(MountainCarEnvCustom, self).__init__()
        self.stepCount = 0
        self.endStep = 500

    def step(self, action):
        self.stepCount += 1
        state, reward, done, _ = super(MountainCarEnvCustom, self).step(action)
        #reward = -1
        reward += state[0] + 0.5
        if done:
            reward += 10
        if not done and self.stepCount > self.endStep:
            done = True
        return state, reward, done, {}

    def reset(self):
        state = super(MountainCarEnvCustom, self).reset()
        self.stepCount = 0
        return state
