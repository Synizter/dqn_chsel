from a2c_agent import ActorCritic
from eeg_env import EEGChannelOptimze
import numpy as np


fname = ['Datasets/Lai_JulyData.mat', 'Datasets/Takahashi_JulyData.mat']
dataset_channel_map = {'F4': 0, 'C4': 1, 'Pa': 2, 'Cz': 3, 'F3': 4, 'C3': 5, 'P3': 6, 'F7': 7, 'T3': 8, 'T5': 9, 
                           'Fp1': 10, 'Fp2': 11, 'T4': 12, 'F8': 13, 'Fz': 14, 'Pz': 15, 'T6': 16, 'O2': 17, 'O1': 18}




def get_expected_return(
    rewards: tf.Tensor, 
    gamma: float, 
    standardize: bool = True) -> tf.Tensor:
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

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

def run_episode(
    initial_state: tf.Tensor,  
    model: tf.keras.Model, 
    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Runs a single episode to collect training data."""

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

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

    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])

    # Apply action to the environment to get next state and reward
    state, reward, done = tf_env_step(action)
    state.set_shape(initial_state_shape)

    # Store reward
    rewards = rewards.write(t, reward)

    if tf.cast(done, tf.bool):
      break

  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()

  return action_probs, values, rewards

if __name__ == "__main__":
    fname = ['Datasets/Takahashi_JulyData.mat']
    dataset_channel_map = {'F4': 0, 'C4': 1, 'P4': 2, 'Cz': 3, 'F3': 4, 'C3': 5, 'P3': 6, 'F7': 7, 'T3': 8, 'T5': 9, 
                           'Fp1': 10, 'Fp2': 11, 'T4': 12, 'F8': 13, 'Fz': 14, 'Pz': 15, 'T6': 16, 'O2': 17, 'O1': 18}
    
    Xx, yy = capilab_dataset2.get(fname)
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise ("NO GPU Available")
    try:
            # Restrict TensorFlow to only use the first GPU
           
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

    env = EEGChannelOptimze(dataset_info, model_set.Custom1DCNN, 0.25)
    num_actions = env.action_space.n  # 2
    num_hidden_units = 128
    model = ActorCritic(num_actions, num_hidden_units)
    eps = np.finfo(np.float32).eps.item()

    seed = 420
    env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

