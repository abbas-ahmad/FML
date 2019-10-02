import numpy as np
import matplotlib.pyplot as plt
import pickle

x = np.arange(1, 301)
x
s = np.random.uniform(-1, 0, 300)


plt.scatter(x, s)
plt.show()


def sigma(x):
    return 1 / (1 + np.exp(-x))


X = np.linspace(-5, 5, 100)
X
plt.plot(X, sigma(X), 'b')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Sigmoid Function')
plt.grid()
plt.text(4, 0.8, r'$\sigma(x)=\frac{1}{1+e^{-x}}$', fontsize=16)
plt.show()

train = pickle.load(open("data/mnist/train.pkl", 'rb'))
train[0]
testX = np.array(train[0][:50000])
testX = np.asfarray(testX)*(0.99/255) + 0.01
y = train[1]
y[y==0]
plt.hist(train[0])
plt.show()
