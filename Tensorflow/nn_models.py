import tensorflow as tf

SEED = 42


class Classifier(tf.keras.Model):
    def __init__(self, input_channels, conv1_out, conv2_out, fc1_units, num_classes):
        super().__init__()
  
        # Convolutional Layers
        self.conv1 = tf.keras.layers.Conv2D(filters=conv1_out,kernel_size= 3,padding='same', activation='relu',
            input_shape=(64,64,input_channels))
        
        self.conv2 = tf.keras.layers.Conv2D(conv2_out, 3,padding='same', activation='relu')

        # Pooling layer
        self.pool =tf.keras.layers.MaxPool2D(2)
  
        # Dropout layers (reduce overfitting)
        self.dropout1 = tf.keras.layers.Dropout(0.15, seed=SEED)
        self.flatten = tf.keras.layers.Flatten()
          # Fully connected layers
        self.fc1 = tf.keras.layers.Dense(fc1_units, activation='relu')

        self.dropout2 = tf.keras.layers.Dropout(0.30, seed=SEED)
        self.fc2 = tf.keras.layers.Dense(num_classes)
  

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout2(x, training=training)
        
        return self.fc2(x)
  
  
