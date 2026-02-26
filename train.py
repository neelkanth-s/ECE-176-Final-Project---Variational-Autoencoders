from keras.datasets import fashion_mnist
import tensorflow
import matplotlib.pyplot as plt
import numpy as np
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape)   # (60000, 28, 28)
print(y_train.shape)   # (60000,)

x_class = x_train[y_train == 0]
fig, axes = plt.subplots(1, 5, figsize=(10,2))

for i, ax in enumerate(axes):
    ax.imshow(x_class[i], cmap="gray")
    ax.axis("off")

plt.show()

X_train_flat = x_class.reshape(len(x_class), 28*28)
print(np.shape(X_train_flat))
mu = np.mean(X_train_flat,0)
var = np.var(X_train_flat,0)
mu = mu.reshape(28,28)
plt.imshow(mu, cmap="gray")
plt.show()