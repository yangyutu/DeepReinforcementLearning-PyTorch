import numpy as np
import math
import random
class HedgingSimulator:

    def __init__(self, config = None, seed = 1):
        self.config = config
        self.stepCount = 0
        self.nbActions = 1
        self.stateDim = 3 # cash position, stock position, the stock price
        self.randomSeed = seed
        self.retHist = []
        self.infoDict = {}

        self.daysInYear = 250
        self.ret_Stock = (self.config['stockGrossReturn'] - 1)/self.daysInYear
        self.ret_Bond = (self.config['bondGrossReturn'] - 1)/self.daysInYear
        self.vol = self.config['stockVol'] / math.sqrt(self.daysInYear)

        self.episodeLength = self.config['episodeLength']

        self.optionStrike = self.config['optionStrike']
        self.S0 = 1
        # risk aversion parameter should be large when vol is small such that signal is large
        self.riskAverse = self.config['riskAverse']
        # reward scale should be adjusted based on actual reward
        self.rewardScale = self.config['rewardScale']


        self.kinkEpisode = 2500
        if 'kinkEpisode' in self.config:
            self.kinkEpisode = self.config['kinkEpisode']

        self.epiCount = -1

    def totalWealth(self):

        res = np.sum(self.currentState[0:2])
        if self.stepCount >= self.episodeLength:
            res += - max(self.currentState[2] - self.optionStrike, 0.0)

        return res

    def rewardCal(self):
        done = False
        reward = 0.0

        if self.stepCount >= self.episodeLength:
            done = True
            # exponenential utility
            # https://en.wikipedia.org/wiki/Exponential_utility
            totalWealth = self.totalWealth()
            reward = (1 - math.exp(-totalWealth * self.riskAverse)) / self.rewardScale
            self.info['totalWealth'] = totalWealth
        return done, reward

    def timeIndexScheduler(self):

        idx = int(self.epiCount / self.kinkEpisode) + 1

        if idx > self.episodeLength:
            return 0
        else:
            return self.episodeLength - idx

    def evolveStock(self, stockValue, nSample = 1):
        return stockValue * np.exp((self.ret_Stock - 0.5 * math.pow(self.vol, 2)) \
                                     + np.random.normal(0, 1, nSample) * self.vol )

    def step(self, action):
        # action means the amount of dollar value stocks to buy
        self.currentState[0] -= action  # reduce cash
        self.currentState[1] += action  # increase stock

        self.currentState[0] = self.currentState[0] * (1 + self.ret_Bond)
        growthFactor = self.evolveStock(1.0)
        self.currentState[1] *= growthFactor
        self.currentState[2] *= growthFactor

        self.stepCount += 1
        self.info['timeStep'] = self.stepCount
        done, reward = self.rewardCal()
        combinedState = {'state': self.currentState.copy(), 'timeStep': self.stepCount}
        self.info['currentState'] = self.currentState.copy()
        self.info['action'] = action
        return combinedState, reward, done, self.info

    def reset(self):
        #self.infoDict['reset'] = True
        self.epiCount += 1
        self.info = {}
        self.stepCount = self.timeIndexScheduler()
        self.info['timeStep'] = self.stepCount
        if self.stepCount == 0:
            self.currentState = np.array([0.0, 0.0, 0.0])
        else:
            self.currentState = np.random.normal(0.0, 1.0, 3)
        #self.currentState[0:2] = np.random.random(2)
        self.S0 = np.random.random() + 0.5
        self.currentState[2] = self.S0
        #self.S0 = 1.0
        #self.currentState[2] = self.S0 * math.exp( (self.ret_Stock - 0.5 * math.pow(self.vol, 2)) * self.stepCount \
        #                                           + np.random.normal() * self.vol * math.sqrt(self.stepCount))


        self.info['initialState'] = self.currentState.copy()
        combinedState = {'state': self.currentState.copy(), 'timeStep': self.stepCount}

        return combinedState
    def close(self):
        pass

    def seed(self):
        pass

    def render(self, mode='human'):
        pass

    def oneStepBenchmark(self, fileName):

        S = np.linspace(0.5, 1.5, 51)
        A = np.linspace(-2, 2, 101)
        #S = [1.0]
        #A = [-1.0, -0.5, 0.0, 0.5, 1.0]
        cash = 0.0
        numSamples = 10000
        result = np.zeros((len(S), len(A)))
        for i, s in enumerate(S):
            stockPrice = s

            for j, a in enumerate(A):
                stockValue = 0.0
                cash = 0.0
                cash -= a  # reduce cash
                stockValue += a  # increase stock

                cash *= (1 + self.ret_Bond)
                growthFactor = self.evolveStock(1.0, nSample = numSamples)
                stockValueVec = stockValue * growthFactor
                stockPriceVec = stockPrice * growthFactor
                totalWealthVec = cash + stockValueVec + - np.maximum(stockPriceVec - self.optionStrike, 0.0)
                rewardVec = (1 - np.exp(-totalWealthVec * self.riskAverse)) / self.rewardScale
                rewardAvg = np.mean(rewardVec)
                result[i, j] = rewardAvg

        np.savetxt(fileName, result, fmt="%.3f")
