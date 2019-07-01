import matplotlib.pyplot as plt 
from random import random
import numpy as np 
#y = x**2 + 1*random(0,1) - 25

def graph():
    x=np.linspace(-5,5,50)
    y=np.ones((50,))
    for i,data in enumerate(x):
        y[i]=data**2 + 1*random() - 25
    plt.figure()
    plt.plot(x, y)
    plt.show()




if __name__ == "__main__":
    graph()