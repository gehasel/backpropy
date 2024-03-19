# %%
import matplotlib.pyplot as plt
import numpy as np
import logging

import backpropy.mnist_for_numpy.mnist as mnist
from backpropy.gradient import Model, BatchGradientDescent, CE, Lin, Softmax, to_column_vector, ReLU

# %%
np.set_printoptions(suppress=True, precision=4)
logging.basicConfig(level=logging.INFO)

# %%
X_train, y_train, X_test, y_test = mnist.load()
# %%
num_datapoints = 500
X_train = X_train[:num_datapoints] / 255.
X_test = X_test / 255.

num = 6
x = X_train[num, :]
y = y_train[num]


# %%
def show_img(x):
    img = x.reshape(28, 28) # First image in the training set.
    plt.imshow(img, cmap='gray')
    plt.show()  # Show the image


show_img(x)
print(f"actual: {y}")

# %%

def one_hot_encode(arr):
    one_hot = np.zeros((arr.size, 10))
    one_hot[np.arange(arr.size), arr] = 1
    return one_hot


y_train_ohe = one_hot_encode(y_train[:num_datapoints])
y_test_ohe = one_hot_encode(y_test)

print(f"{X_train.shape=}")
print(f"{y_train_ohe.shape=}")

# %%
# display(X_train[:5, :])
# display(y_train_ohe[:5])

x_cv = to_column_vector(x)
y_cv = to_column_vector(y_train_ohe[num])
# %%

rng = np.random.default_rng(seed=123)

lower_rand = -0.1
upper_rand = 0.1
in1 = 784
out1 = 300
z_1 = Lin(rng.uniform(low=lower_rand, high=upper_rand, size=(in1, out1)), rng.uniform(low=lower_rand, high=upper_rand, size=(out1, 1)), name='z_1')
f_1 = ReLU(name='f_1')

out2 = 100
z_2 = Lin(rng.uniform(low=lower_rand, high=upper_rand, size=(out1, out2)), rng.uniform(low=lower_rand, high=upper_rand, size=(out2, 1)), name='z_2')
f_2 = ReLU(name='f_2')

out3 = 10
z_3 = Lin(rng.uniform(low=lower_rand, high=upper_rand, size=(out2, out3)), rng.uniform(low=lower_rand, high=upper_rand, size=(out3, 1)), name='z_3')
f_3 = Softmax(name='f_3')

loss = CE()  # is CE the right loss here? Do I have to transform the final output? right now, the real y and y_pred are vectors of shape (10, 1)

myoptimizer = BatchGradientDescent(batch_size=10, learning_rate=0.1, epochs=5)
mymodel = Model(
    [z_1, f_1, z_2, f_2, z_3, f_3],
    loss,
    X_train,
    y_train_ohe,
    myoptimizer,
)
# %%
mymodel.fit()
# %%
# some visual testing:
for test_num in range(20):
    x_test1 = X_test[test_num]
    show_img(x_test1)
    print(f"{np.argmax(mymodel.predict(x_test1))} with confidence {np.max(mymodel.predict(x_test1)):.4f}")
    print(f"second best guess: {np.argsort(np.max(mymodel.predict(x_test1), axis=1))[-2]} with confidence {np.sort(np.max(mymodel.predict(x_test1), axis=1))[-2]:.4f}")
    print(f"actual: {y_test[test_num]}")
# %%
from backpropy.gradient import accuracy
accuracy(y_test, np.array([np.argmax(mymodel.predict(x)) for x in X_test]))
# %%
