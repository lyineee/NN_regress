import matplotlib.pyplot as plt 
from random import random

import numpy as np 

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

#y = x**2 + 1*random(0,1) - 25

def get_train_data(n):
    feature=np.linspace(-5,5,n)
    target=np.ones((n,))
    for i,data in enumerate(feature):
        target[i]=data**2 + 1*random() - 25
    return feature,target

def build_model():
    model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu,input_shape=(1,)),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()

    feature,target=get_train_data(10000)
    val_feature,val_target=get_train_data(1000)
    history = model.fit(
      feature, target,epochs=5,
      validation_data=(val_feature,val_target))

    model.predict([1])
    x=np.linspace(-5,5,100)
    pred_y=model.predict(x)
    plt.figure()
    plt.plot(x, pred_y)
    plt.show()
    