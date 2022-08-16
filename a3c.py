from typing import Any, List, Sequence, Tuple

import silence_tensorflow #うるさいしないために
silence_tensorflow.silence_tensorflow()

from a2c_agent import ActorCritic
from eeg_env import EEGChannelOptimze
import numpy as np
import tensorflow as tf
import numpy as np
import model_set
import capilab_dataset2
import statistics

from sklearn.model_selection import train_test_split

# Hepler Func--------------------------------------------------------------------------
def env_step(action) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""
  state, reward, done, _ = env.step(action)
  return (state.astype(np.int32), np.array(reward, np.float32), np.array(done, np.bool))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action], [tf.int32, tf.float32, tf.bool])


def get_expected_return(rewards: tf.Tensor, gamma: float, standardize: bool = True) -> tf.Tensor:
  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)  #place holder for expect return calculation

  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = ((returns - tf.math.reduce_mean(returns)) / 
               (tf.math.reduce_std(returns) + eps))

  return returns


def compute_loss(action_probs: tf.Tensor,  values: tf.Tensor,  returns: tf.Tensor) -> tf.Tensor:
  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss

def run_episode(initial_state: tf.Tensor,  model: tf.keras.Model, max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state

  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)

    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)
    # Sample next action from the action probability distribution

    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)


    state, reward, done = tf_env_step(action)

    state.set_shape(initial_state_shape)

    # Store critic values
    values = values.write(t, tf.squeeze(value))
    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])
    # Apply action to the environment to get next state and reward
    # Store reward
    rewards = rewards.write(t, reward)

    if done:
      break

  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()

  return action_probs, values, rewards


instance_counter = 1

@tf.function
def train_step(initial_state: tf.Tensor, model: tf.keras.Model, gamma: float, max_steps_per_episode: int):
  """Runs a model training step."""
  global instance_counter
  with tf.GradientTape() as tape:
    # Run the model for one episode to collect training data
    action_probs, values, rewards = run_episode(initial_state, model, max_steps_per_episode) 
    
    # Calculate expected returns

    returns = get_expected_return(rewards, gamma)
    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 
    # Calculating loss values to update our network
    loss = compute_loss(action_probs, values, returns)

  # Compute the gradients from the loss

  grads = tape.gradient(loss, model.trainable_variables)
  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward, loss

# init --------------------------------------------------------------------------------

fname = ['Datasets/Lai_JulyData.mat',  'Datasets/Sugiyama_JulyData.mat']
dataset_channel_map = {'F4': 0, 'C4': 1, 'Pa': 2, 'Cz': 3, 'F3': 4, 'C3': 5, 'P3': 6, 'F7': 7, 'T3': 8, 'T5': 9, 
                           'Fp1': 10, 'Fp2': 11, 'T4': 12, 'F8': 13, 'Fz': 14, 'Pz': 15, 'T6': 16, 'O2': 17, 'O1': 18}

Xx, yy = capilab_dataset2.get(fname)

dataset_info = None
try:
        # Restrict TensorFlow to only use the first GPU
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

    X, x_test,  y, y_test = train_test_split(Xx, yy, test_size = .1, stratify = yy, random_state = 420)
    dataset_info = {
        "X":X,
        "y":y,
        "x_test":x_test,
        "y_test":y_test,
        "nbr_class":y.shape[1],
        "data_shape":(X.shape[1], X.shape[2]),
        "nbr_data":X.shape[0],
        "ch_map":dataset_channel_map
    }
except Exception as e:
    print(e)


if dataset_info is None:
    print("Error")
else:
    print("Creating an evnironment...")
    env = EEGChannelOptimze(dataset_info, model_set.Custom1DCNN, 0.25)
    print("Done")
    
import collections

seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

eps = np.finfo(np.float32).eps.item()

#Actor Critic Model
num_actions = env.action_space.n  # 2
num_hidden_units = 128
model = ActorCritic(num_actions, num_hidden_units)
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


#Episode Parameter
min_episodes_criterion = 15
max_episodes = 300
max_steps_per_episode = 6

reward_threshold = 0.75
running_reward = 0

# Discount factor for future rewards
gamma = 0.99
# Keep last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)
episodes_loss: collections.deque = collections.deque(maxlen=min_episodes_criterion)


# MAIN--------------------------------------------------------------------------------------

import tqdm
with tqdm.trange(max_episodes) as t:
    for i in t:
        initial_state = tf.constant(env.reset(), dtype=tf.int32) #reset env
        episode_reward,loss = train_step(initial_state, model, gamma, max_steps_per_episode)
        episode_reward = float(episode_reward)
        loss = float(loss)
        
        episodes_reward.append(episode_reward)
        episodes_loss.append(loss)
        running_reward = statistics.mean(episodes_reward)
        running_loss = statistics.mean(episodes_loss)

        t.set_description(f'Episode {i}')
        t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

 
        with open('log_reward_loss.txt', 'a+') as f:
            log = '{},{:.3f},{:.3f},{}\n'.format(i,episode_reward, loss, env.state.tolist())
            f.write(log)
        tf.keras.models.save_model(model, 'a2c')
        
        if running_reward > reward_threshold and i >= min_episodes_criterion:  
            break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')