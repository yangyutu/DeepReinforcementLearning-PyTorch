import numpy as np
import math
import random
class HedgingSimulator:

    def __init__(self, config = None, seed = 1):
        self.config = config
        self.stepCount = 0
        self.currentState = 0.0
        self.nbActions = 1
        self.stateDim = 3 # cash position, stock position, the stock price
        self.endStep = 30
        self.cumRet = 1.0
        self.randomSeed = seed
        self.retHist = []
        self.AR1Coeff = self.config['AR1Coeff']

        self.nSample = 200
        if 'nSample' in self.config:
            self.nSample = self.config['nSample']

        self.infoDict = {}
        self.episodeLength = self.endStep
        self.ret_Stock = math.pow((1.1 - 1), 1/250)
        self.ret_Bond = math.pow((1.00 - 1), 1/250)
        self.vol = 0.3 / math.sqrt(250)

        self.optionStrike = 1

    def totalWealth(self):
        return np.sum(self.currentState)

    def rewardCal(self):
        done = False
        reward = 0.0

        if self.stepCount >= self.episodeEndStep:
            done = True
            # exponenential utility
            # https://en.wikipedia.org/wiki/Exponential_utility
            reward = (1 - math.exp(-self.totalWealth()))
        return done, reward

    def timeIndexScheduler(self):

        kink = 2500

        idx = int(self.epiCount / kink)

        if idx > self.episodeEndStep:
            return 0
        else:
            return random.randint(self.episodeEndStep - 1 - idx, self.episodeEndStep - 1)

    def evolveStock(self, stockValue):
        # generate n samples of future values
        return stockValue * math.exp(self.ret_Stock - 0.5 * math.pow(self.vol, 2) + np.random.normal() * self.vol)

    def step(self, action):
        # action means the amount of dollar value stocks to buy
        self.currentState[0] -= action  # reduce cash
        self.currentState[1] += action  # increase stock

        self.currentState[0] = self.currentState[0] * (1 + self.ret_Bond)
        self.currentState[1] = self.evolveStock(self.currentState[1])
        self.currentState[2] = self.evolveStock(self.currentState[2])

        self.stepCount += 1
        self.info['timeStep'] = self.stepCount
        done, reward = self.rewardCal()
        combinedState = {'state': self.currentState.copy(), 'timeStep': self.stepCount}

        return combinedState, reward, done, self.infoDict

    def reset(self):
        #self.infoDict['reset'] = True
        self.stepCount = self.timeIndexScheduler()
        self.info['timeStep'] = self.stepCount

        self.currentState = np.array([0, 0, 0])
        self.currentState[0:2] = np.random.normal(2)
        self.currentState[2] = self.S0 * math.exp( (self.ret_Stock - 0.5 * math.pow(self.vol, 2)) * self.episodeLength \
                                                   + np.random.normal() * self.vol * math.sqrt(self.episodeLength))



        combinedState = {'state': self.currentState.copy(), 'timeStep': self.stepCount}

        return combinedState
    def close(self):
        pass

    def seed(self):
        pass

    def render(self, mode='human'):
        pass