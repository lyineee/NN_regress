import matplotlib.pyplot as plt 
from random import random

import numpy as np 

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

#y = x**2 + 1*random(0,1) - 25

def graph(n):
    x=np.linspace(-5,5,n)
    y=np.ones((n,))
    for i,data in enumerate(x):
        y[i]=x[i]**2 + 1*random() - 25
    plt.figure()
    plt.plot(x, y)
    plt.show()


def add_layer(inputs,in_size,out_size,activation_function=None):
    #define weights
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    #define biases
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    #construct a neuron
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    #use activation function
    if activation_function == None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


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

    feature,target=get_train_data(20000)
    history = model.fit(
      feature, target)

    model.predict([1])
    x=np.linspace(-5,5,100)
    pred_y=model.predict(x)
    plt.figure()
    plt.plot(x, pred_y)
    plt.show()