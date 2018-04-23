import argparse
import gym
from gym.spaces import Box, Discrete
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape, merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, l1
from keras.constraints import maxnorm, unitnorm
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import theano.tensor as T
import numpy as np
from buffer import Buffer
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='NAF')
parser.add_argument('--batch_size', nargs='?', type=int, default=100)
parser.add_argument('--hidden_size', nargs='?', type=int, default=100)
parser.add_argument('--layers', nargs='?', type=int, default=2)
parser.add_argument('--batch_norm', action="store_true", default=False)
parser.add_argument('--no_batch_norm', action="store_false", dest="batch_norm")
parser.add_argument('--max_norm', nargs='?', type=int)
parser.add_argument('--unit_norm', action='store_true', default=False)
parser.add_argument('--l2_reg', nargs='?', type=float)
parser.add_argument('--l1_reg', nargs='?', type=float)
parser.add_argument('--replay_size', nargs='?', type=int, default=100000)
parser.add_argument('--train_repeat', nargs='?', type=int, default=10)
parser.add_argument('--gamma', nargs='?', type=float, default=0.99)
parser.add_argument('--tau', nargs='?', type=float, default=0.001)
parser.add_argument('--episodes', nargs='?', type=int, default=500)
parser.add_argument('--max_timesteps', nargs='?', type=int, default=200)
parser.add_argument('--activation', nargs='?', choices=['tanh', 'relu'], default='tanh')
parser.add_argument('--optimizer', nargs='?', choices=['adam', 'rmsprop'], default='adam')
parser.add_argument('--optimizer_lr', nargs='?', type=float, default=0.001)
parser.add_argument('--noise', nargs='?', choices=['linear_decay', 'exp_decay', 'fixed', 'covariance'], default='linear_decay')
parser.add_argument('--noise_scale', nargs='?', type=float, default=0.01)
parser.add_argument('--display', action='store_true', default=False)
parser.add_argument('--no_display', dest='display', action='store_false')
parser.add_argument('--gym_record')
parser.add_argument('environment')
args = parser.parse_args()

assert K._BACKEND == 'theano', "only works with Theano as backend"

# create environment
env = gym.make(args.environment).env
assert isinstance(env.observation_space, Box), "observation space must be continuous"
assert isinstance(env.action_space, Box), "action space must be continuous"
assert len(env.action_space.shape) == 1
num_actuators = env.action_space.shape[0]
print("num_actuators:", num_actuators)

# start monitor for OpenAI Gym
if args.gym_record:
  env.monitor.start(args.gym_record)

# optional norm constraint
if args.max_norm:
  kernel_constraint = maxnorm(args.max_norm)
elif args.unit_norm:
  kernel_constraint = unitnorm()
else:
  kernel_constraint = None

# optional regularizer
def regularizer():
  if args.l2_reg:
    return l2(args.l2_reg)
  elif args.l1_reg:
    return l1(args.l1_reg)
  else:
    return None

# helper functions to use with layers
if num_actuators == 1:
  # simpler versions for single actuator case
  def _L(x):
    return K.exp(x)

  def _P(x):
    return x**2

  def _A(t):
    m, p, u = t
    return -(u - m)**2 * p

  def _Q(t):
    v, a = t
    return v + a
else:
  # use Theano advanced operators for multiple actuator case
  def _L(x):
    # initialize with zeros
    batch_size = x.shape[0]
    a = T.zeros((batch_size, num_actuators, num_actuators))
    # set diagonal elements
    batch_idx = T.extra_ops.repeat(T.arange(batch_size), num_actuators)
    diag_idx = T.tile(T.arange(num_actuators), batch_size)
    b = T.set_subtensor(a[batch_idx, diag_idx, diag_idx], T.flatten(T.exp(x[:, :num_actuators])))
    # set lower triangle
    cols = np.concatenate([np.array(range(i), dtype=np.uint) for i in range(num_actuators)])
    rows = np.concatenate([np.array([i]*i, dtype=np.uint) for i in range(num_actuators)])
    cols_idx = T.tile(T.as_tensor_variable(cols), batch_size)
    rows_idx = T.tile(T.as_tensor_variable(rows), batch_size)
    batch_idx = T.extra_ops.repeat(T.arange(batch_size), len(cols))
    c = T.set_subtensor(b[batch_idx, rows_idx, cols_idx], T.flatten(x[:, num_actuators:]))
    return c

  def _P(x):
    return K.batch_dot(x, K.permute_dimensions(x, (0,2,1)))

  def _A(t):
    m, p, u = t
    d = K.expand_dims(u - m, -1)
    return -K.batch_dot(K.batch_dot(K.permute_dimensions(d, (0,2,1)), p), d)

  def _Q(t):
    v, a = t
    return v + a

# helper function to produce layers twice
def createLayers():
  x = Input(shape=env.observation_space.shape, name='x')
  u = Input(shape=env.action_space.shape, name='u')
  if args.batch_norm:
    h = BatchNormalization()(x)
  else:
    h = x
  for i in range(args.layers):
    h = Dense(args.hidden_size, activation=args.activation, name='h'+str(i+1),
        kernel_constraint=kernel_constraint, kernel_regularizer=regularizer())(h)
    if args.batch_norm and i != args.layers - 1:
      h = BatchNormalization()(h)
  v = Dense(1, name='v', kernel_constraint=kernel_constraint, kernel_regularizer=regularizer())(h)
  m = Dense(num_actuators, name='m', kernel_constraint=kernel_constraint, kernel_regularizer=regularizer())(h)
  l0 = Dense(num_actuators * (num_actuators + 1) // 2, name='l0',
        kernel_constraint=kernel_constraint, kernel_regularizer=regularizer())(h)
  l = Lambda(_L, output_shape=(num_actuators, num_actuators), name='l')(l0)
  p = Lambda(_P, output_shape=(num_actuators, num_actuators), name='p')(l)
  a = Lambda(_A, output_shape=(num_actuators,), name="a")([m, p, u])
  q = Lambda(_Q, output_shape=(num_actuators,), name="q")([v, a])
  return x, u, m, v, q, p, a

x, u, m, v, q, p, a = createLayers()

# wrappers around computational graph
fmu = K.function([K.learning_phase(), x], m)
mu = lambda x: fmu([0, x])

fP = K.function([K.learning_phase(), x], p)
P = lambda x: fP([0, x])

fA = K.function([K.learning_phase(), x, u], a)
A = lambda x, u: fA([0, x, u])

fQ = K.function([K.learning_phase(), x, u], q)
Q = lambda x, u: fQ([0, x, u])

# main model
model = Model(inputs=[x,u], outputs=q)
model.summary()

if args.optimizer == 'adam':
  optimizer = Adam(args.optimizer_lr)
elif args.optimizer == 'rmsprop':
  optimizer = RMSprop(args.optimizer_lr)
else:
  assert False
model.compile(optimizer=optimizer, loss='mse')

# another set of layers for target model
x, u, m, v, q, p, a = createLayers()

# V() function uses target model weights
fV = K.function([K.learning_phase(), x], v)
V = lambda x: fV([0, x])

# target model is initialized from main model
target_model = Model(inputs=[x,u], outputs=q)
target_model.set_weights(model.get_weights())

# replay memory
R = Buffer(args.replay_size, env.observation_space.shape, env.action_space.shape)

# the main learning loop
total_reward = 0
reward_list = []

for i_episode in range(args.episodes):
    observation = env.reset()
    #print "initial state:", observation
    episode_reward = 0
    for t in range(args.max_timesteps):
        if args.display:
          env.render()

        # predict the mean action from current observation
        x = np.array([observation])
        u = mu(x)[0]

        # add exploration noise to the action
        if args.noise == 'linear_decay':
          action = u + np.random.randn(num_actuators) / (i_episode + 1)
        elif args.noise == 'exp_decay':
          action = u + np.random.randn(num_actuators) * 10 ** -i_episode
        elif args.noise == 'fixed':
          action = u + np.random.randn(num_actuators) * args.noise_scale
        elif args.noise == 'covariance':
          if num_actuators == 1:
            std = np.minimum(args.noise_scale / P(x)[0], 1)
            #print "std:", std
            action = np.random.normal(u, std, size=(1,))
          else:
            cov = np.minimum(np.linalg.inv(P(x)[0]) * args.noise_scale, 1)
            #print "covariance:", cov
            action = np.random.multivariate_normal(u, cov)
        else:
          assert False
        #print "action:", action, "Q:", Q(x, np.array([action])), "V:", V(x)
        #print "action:", action, "advantage:", A(x, np.array([action]))
        #print "mu:", u, "action:", action
        #print "Q(mu):", Q(x, np.array([u])), "Q(action):", Q(x, np.array([action]))

        # take the action and record reward
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        #print "reward:", reward
        #print "poststate:", observation

        # add experience to replay memory
        R.add(x[0], action, reward, observation, done)

        loss = 0
        # perform train_repeat Q-updates
        for k in range(args.train_repeat):
          preobs, actions, rewards, postobs, terminals = R.sample(args.batch_size)

          # Q-update
          v = V(postobs)
          y = rewards + args.gamma * np.squeeze(v)
          loss += model.train_on_batch([preobs, actions], y)

          # copy weights to target model, averaged by tau
          weights = model.get_weights()
          target_weights = target_model.get_weights()
          for i in range(len(weights)):
            target_weights[i] = args.tau * weights[i] + (1 - args.tau) * target_weights[i]
          target_model.set_weights(target_weights)
        #print "average loss:", loss/k

        if done:
            break

    print("Episode {} finished after {} timesteps, reward {}".format(i_episode + 1, t + 1, episode_reward))
    total_reward += episode_reward
    reward_list.append(episode_reward)

print("Average reward per episode {}".format(total_reward / args.episodes))

plt.plot(range(0,args.episodes), reward_list, linewidth=2)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Performance")
plt.savefig('naf-'+args.environment+"-"+str(args.episodes)+'-'+str(args.max_timesteps)+'.jpg')
plt.close()

if args.gym_record:
  env.monitor.close()
