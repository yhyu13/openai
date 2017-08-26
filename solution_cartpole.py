'''
author: Thyrix Yang
github: https://github.com/ThyrixYang
'''



import numpy as np 
import gym

# Cross Entropy Method

# A simple evolutional RL method
# The core idea is to maintain a distribution of a set of weights(of a model),
# then try weights produced by this distribution and select the best.
# Won't work on complex models. I used only four weights in this model.

# paper:  [1] http://iew3.technion.ac.il/CE/files/papers/Learning%20Tetris%20Using%20the%20Noisy%20Cross-Entropy%20Method.pdf
#         [2] https://papers.nips.cc/paper/5190-approximate-dynamic-programming-finally-performs-well-in-the-game-of-tetris.pdf

class CEMOptimizer:

  def __init__(self, weights_dim, batch_size=1000, deviation=100, rho=0.1, eta=0.1, mean=None):
    self.rho = rho
    self.eta = eta
    self.weights_dim = weights_dim
    self.mean = mean if mean!=None else np.zeros(weights_dim)
    self.deviation = np.full(weights_dim, deviation)
    self.batch_size = batch_size
    self.select_num = int(batch_size * rho)

    assert(self.select_num > 0)

  def update_weights(self, weights, rewards):
    rewards = np.array(rewards)
    weights = np.array(weights)
    sorted_idx = (-rewards).argsort()[:self.select_num]
    top_weights = weights[sorted_idx]
    self.mean = np.sum(top_weights, axis=0) / self.select_num
    self.deviation = np.std(top_weights, axis=0)

    assert(len(self.deviation)==self.weights_dim)

  def sample_batch_weights(self):
    return [np.random.normal(self.mean, self.deviation + self.eta) \
        for _ in range(self.batch_size)]

  def get_weights(self):
    return self.mean


def train_cartpole():

  def select_action(ob, weights):
    w = np.reshape(weights, (1, 4))
    ob = np.reshape(ob, (4, 1))
    def sigmoid(x):
      return 1 / (1 + np.exp(-x))
    v = sigmoid(np.dot(w, ob))
    return 0 if v < 0.5 else 1

  opt = CEMOptimizer(4*1, 40)
  env = gym.make("CartPole-v0")
  env = gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
  epoch = 10

  def test():
    W = opt.get_weights()
    observation = env.reset()
    accreward = 0
    while True:
      env.render()
      action = select_action(observation, W)
      observation, reward, done, info = env.step(action)
      accreward += reward
      if done:
        print("test end with reward: {}".format(accreward))
        break

  for ep in range(epoch):
    print("start epoch {}".format(ep))
    weights = opt.sample_batch_weights()
    rewards = []
    for b in range(opt.batch_size):
      observation = env.reset()
      accreward = 0
      while True:
        action = select_action(observation, weights[b])
        observation, reward, done, info = env.step(action)
        accreward += reward
        if done:
          break
      rewards.append(accreward)
    opt.update_weights(weights, rewards)
    test()

if __name__ == '__main__':
  train_cartpole()
