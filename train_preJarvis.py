import gym
import sys
import json
import oandapyV20
import numpy as np
import pandas as pd

from api import *
from csv import reader
from os.path import join
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
from stable_baselines import A2C, ACKTR, PPO2

from env.BitcoinTradingEnv import BitcoinTradingEnv
# from env.traderEnv import BitcoinTradingEnv
from util.indicators import add_indicators
from util.plots import plotEveryReward


curr_idx = -1
num_train_iterations = 2
reward_strategy = 'sortino'

from_ = '2017-01-01T00:00:00Z'
to = '2017-06-30T00:00:00Z'
gran = 'H4'
instr='EUR_USD'

access_token = "3282a2c5c3e6e10c06ae7d04f365ae25-efe560b3801a46ee6ad6fe781632a78e"
accountID = "101-004-12211600-001"


api = oandapyV20.API(access_token=access_token)

df = requestCandles(api, gran, from_, to, instr)

saveDf = True
filename = join('data', '{}.{}.out'.format(instr, gran))

if saveDf:
    df.to_csv(filename)


test_len = int(len(df) * 0.2)
train_len = int(len(df)) - test_len

train_df = df[:train_len]
test_df = df[train_len:]

# ====== ENVIRONMENT SETUP =======
trainEnv = DummyVecEnv([lambda: BitcoinTradingEnv(
        train_df)])

testEnv = DummyVecEnv([lambda: BitcoinTradingEnv(
        test_df)])


model_params = {
    'n_steps': 243,
    'gamma': 0.94715,
    'learning_rate': 0.00157,
    'ent_coef': 2.29869,
    'cliprange':  0.38388,
    'noptepochs': 35,
    'lam': 0.89837,
}

# This is stupid
if curr_idx == -1:
    model = PPO2(MlpLnLstmPolicy, trainEnv, verbose=0, nminibatches=1,
            tensorboard_log="./tensorboard", **model_params)
else:
    model = PPO2.load('./agents/ppo2_' + reward_strategy + '_' + str(curr_idx) + '.pkl', env=trainEnv)

# ====== TRAIN THE MODEL ======
rewards = [[] for i in range(num_train_iterations)]

for idx in range(curr_idx + 1, num_train_iterations):
    print('[', idx, '] Training for: ', train_len, ' time steps')

    model.learn(total_timesteps=train_len)

    obs = testEnv.reset()
    done, reward_sum = False, 0

    while not done:
        action, _states = model.predict(obs)
        # todo - Make magic to predict the next action ?

        obs, reward, done, info = testEnv.step(action)
        reward_sum += reward
        rewards[idx].append(reward_sum[0])

    print('[', idx, '] Total reward: ', reward_sum, ' (' + reward_strategy + ')')
    model.save(join('agents', 'ppo2_' + reward_strategy + '_' + str(idx) + '.pkl'))
    print('Finished')
    # plotEveryReward(rewards, join('plots', 'reward-plot-train'), label='Cumulative Rewards')

