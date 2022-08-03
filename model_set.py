#@title Custom Model 
#Network define
import tensorflow as tf

class Custom1DCNN(tf.keras.Model):
    def __init__(self, input_shape = (100,19), output_classes = 3, name="1D CNN", **kwarg):
        super(Custom1DCNN, self).__init__()

        #layer definition
        self.kernel_size_0 = 20
        self.kernel_size_1 = 6
        self.drop_rate = .5
        self.pool_size = 2
        self.in_shape = input_shape
        #layer 1
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=self.kernel_size_0, padding="same", activation="relu", input_shape=input_shape)
        self.batch1 = tf.keras.layers.BatchNormalization()
        #layer2
        self.conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=self.kernel_size_0, padding="valid", activation="relu")
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.spatial_drop2 = tf.keras.layers.SpatialDropout1D(self.drop_rate)
        #layer3
        self.conv3 = tf.keras.layers.Conv1D(filters=32, kernel_size=self.kernel_size_1, padding="valid", activation="relu")
        self.avg3 = tf.keras.layers.AveragePooling1D(pool_size=self.pool_size)
        #layer 4
        self.conv4 = tf.keras.layers.Conv1D(filters = 32, kernel_size=self.kernel_size_1, padding="valid", activation="relu")
        self.spatial_drop4 = tf.keras.layers.SpatialDropout1D(self.drop_rate)

        #OUT
        self.flatten = tf.keras.layers.Flatten() 
        self.denseout_1  = tf.keras.layers.Dense(296, activation="relu")
        self.dropout_1 = tf.keras.layers.Dropout(self.drop_rate)
        self.denseout_2  = tf.keras.layers.Dense(148, activation="relu")
        self.dropout_2 = tf.keras.layers.Dropout(self.drop_rate)
        self.denseout_3 = tf.keras.layers.Dense(74, activation="relu")
        self.dropout_3 = tf.keras.layers.Dropout(self.drop_rate)
        self.out = tf.keras.layers.Dense(output_classes, activation='softmax')
    
    def call(self, input_tensor):
        cv1 = self.conv1(input_tensor)
        bn1 = self.batch1(cv1)
        cv2 = self.conv2(bn1)
        bn2 = self.batch2(cv2)
        sd2 = self.spatial_drop2(bn2)
        cv3 = self.conv3(sd2)
        av3 = self.avg3(cv3)
        cv4 = self.conv4(av3)
        sd4 = self.spatial_drop4(cv4)
        fl  = self.flatten(sd4)
        dn1 = self.denseout_1(fl)
        do1 = self.dropout_1(dn1)
        dn2 = self.denseout_2(do1)
        do2 = self.dropout_2(dn2)
        dn3 = self.denseout_3(do2)

        return self.out(dn3)
    
    def model(self):
        x = tf.keras.layers.Input(shape=self.in_shape)
        return tf.keras.Model(inputs=[x],  outputs=self.call(x))

if __name__ == "__main__":
    model = Custom1DCNN()
    model.model().summary()
