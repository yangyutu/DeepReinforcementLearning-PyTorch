

from Env.CustomEnv.DeepHedging.HedgingEnv import HedgingSimulator
import json

configName = 'config.json'
with open(configName ,'r') as f:
    config = json.load(f)

env = HedgingSimulator(config)


env.oneStepBenchmark('test.txt')