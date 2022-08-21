from typing import Any, List, Sequence, Tuple

import silence_tensorflow #うるさいしないために
silence_tensorflow.silence_tensorflow()

import tensorflow as tf
import os

class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""
    def __init__(self, num_actions: int, num_hidden_units: int):
        super().__init__()

        self.common = tf.keras.layers.Dense(num_hidden_units, activation="relu")
        self.actor = tf.keras.layers.Dense(num_actions)
        self.critic = tf.keras.layers.Dense(1)
        
        self.loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)


if __name__ == "__main__":

    
    if os.path.isdir('a2c'):
        print("loading checkpoint")
        agent = tf.keras.models.load_model('a2c')
    else:
        print("Create new model")
        agent = ActorCritic(19, 128)     
        
    

