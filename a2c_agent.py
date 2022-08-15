import tensorflow as tf

class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""
    def __init__(self, num_actions: int, num_hidden_units: int):
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)
        
        self.loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)

    def compute_loss(self, action_probs: tf.Tensor,  values: tf.Tensor,  returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = huber_loss(values, returns)

        return actor_loss + critic_loss

    def run_episode(self, env):
        action_probs = tf.TensorArray(dtype=tf.float32, size = 0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size = 0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size = 0, dynamic_size=True)

        
    
    def train(self):
        with tf.GradientTape() as gd:
            actions_probs, value


