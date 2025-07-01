import numpy as np

X = np.array([1,2,3,4,5])
Y = 0.5*X

w = np.random.rand()

epochs = 1000
learning_rate = 0.01

for epoch in range(epochs):
    for x, y_real in zip(X,Y):
        y_pred = x*w
        error = y_pred - y_real
        loss = 0.5*(error**2)

        gradient = error * x
        w =  w - gradient

    if epochs%1 ==0:
        print("Época:", epoch, "w:", w)

print(w)