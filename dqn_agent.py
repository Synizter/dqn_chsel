
import silence_tensorflow
silence_tensorflow.silence_tensorflow()
import tensorflow as tf

class DQNAgent(tf.keras.Model):
    def __init__(self, action_space:int, num_hidden_units: int):
        super().__init__()
        self.action_space = action_space
        self.d1 = tf.keras.layers.Dense(num_hidden_units // 2, activation = 'relu', kernel_initializer = 'he_uniform')
        self.d2 = tf.keras.layers.Dense(num_hidden_units, activation = 'relu', kernel_initializer = 'he_uniform')
        self.out = tf.keras.layers.Dense(action_space, activation = 'linear', kernel_initializer = 'he_uniform')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.d1(inputs)
        x = self.d2(x)
        x = self.out(x)
        return x
    
    def model(self, inputs_shape):
        inputs = tf.keras.layers.Input(shape = inputs_shape)
        return tf.keras.Model(inputs = inputs, outputs = self.call(inputs))

if __name__ == "__main__":
    
    agent= DQNAgent(19, 128)
    import numpy as np
    inp = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    inp = tf.expand_dims(inp, 0)
    
    out1 = agent.predict(inp) #np.out
    out2 = agent(inp, training = False) #tenosr shape out

    action = tf.argmax(out2, axis = 1).numpy()
    print(action)
    print(out2)