import oandapyV20
import numpy as np
import pandas as pd

from os.path import join
from stable_baselines import A2C, ACKTR, PPO2
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from oandapyV20.contrib.factories import InstrumentsCandlesFactory

from src.env.BitcoinTradingEnv import BitcoinTradingEnv
from src.api import *
# from src.env.traderEnv import BitcoinTradingEnv
from src.util.indicators import add_indicators
from src.util.plots import plotEveryReward



'''
from src.libs.balancing import *
from src.libs.treatment import *
from src.libs.evaluate import *
from src.args import *
'''

class Puppet:
    def __init__(self, args, debug, outputDir):
        """ I could pass everything from args to self right here to allow better interpretation
        of what parameters are needed fo use this class, but since we're using JARVIS that information
        can be seen in 'args.py' """
        self.args = args
        self.debug = debug
        self.outputDir = outputDir
        '''
        if 'classifierParams' in self.args.keys() and self.args['classifierParams'] is not None:
            self.clf = self.args['classifier'](**self.args['classifierParams'])
        else:
            self.clf = self.args['classifier']()
        '''
    def pipeline(self):
        api = oandapyV20.API(access_token=self.args['accessToken'])
        curr_idx = -1

        df = requestCandles(api, self.args['gran'], self.args['from'], self.args['to'], self.args['instr'])

        # fixme
        saveDf = True
        filename = join('src', 'data', '{}.{}.out'.format(self.args['instr'], self.args['gran']))

        if saveDf:
            df.to_csv(filename)

        test_len = int(len(df) * 0.2)
        train_len = int(len(df)) - test_len

        train_df = df[:train_len]
        test_df = df[train_len:]

        # ====== ENVIRONMENT SETUP =======
        trainEnv = DummyVecEnv([lambda: BitcoinTradingEnv(train_df)])

        testEnv = DummyVecEnv([lambda: BitcoinTradingEnv(test_df)])

        model_params = {
            'n_steps': 243,
            'gamma': 0.94715,
            'learning_rate': 0.00157,
            'ent_coef': 2.29869,
            'cliprange': 0.38388,
            'noptepochs': 35,
            'lam': 0.89837,
        }

        # This is stupid
        if curr_idx == -1:
            model = PPO2(MlpLnLstmPolicy, trainEnv, verbose=0, nminibatches=1,
                         tensorboard_log=self.outputDir, **model_params)
        else:
            model = PPO2.load(join(self.outputDir,'ppo2_' + self.args['rewardStrategy'] + '_' + str(curr_idx) + '.pkl'),
                              env=trainEnv)

        # ====== TRAIN THE MODEL ======
        rewards = [[] for i in range(self.args['numTrainIterations'])]

        for idx in range(curr_idx + 1, self.args['numTrainIterations']):
            print('[', idx, '] Training for: ', train_len, ' time steps')

            model.learn(total_timesteps=train_len)

            print('[', idx, '] Testing...')
            obs = testEnv.reset()
            done, reward_sum = False, 0

            while not done:
                action, _states = model.predict(obs)
                # todo - Make magic to predict the next action ?
                print('action: {}'.format(action))
                obs, reward, done, info = testEnv.step(action)
                reward_sum += reward
                print('reward: {}'.format(reward))
                rewards[idx].append(reward_sum[0])

            print('[', idx, '] Total reward: ', reward_sum, ' (' + self.args['rewardStrategy'] + ')')
            print(rewards)
            model.save(join(self.outputDir, 'ppo2_' + self.args['rewardStrategy'] + '_' + str(idx) + '.pkl'))
        plotEveryReward(rewards, join(self.outputDir, 'reward-plot-train'), label='Cumulative Rewards')
