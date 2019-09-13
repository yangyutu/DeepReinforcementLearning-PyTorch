import json
import numpy as np
from Env.CustomEnv.TwoArmRobot.TwoArmRobotEnv import TwoArmEnvironmentContinuous

configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)
env = TwoArmEnvironmentContinuous(config)


state = env.reset()



for i in range(100):
    action = np.random.randn(2)
    env.step(action)
    print(env.effectorPosition)