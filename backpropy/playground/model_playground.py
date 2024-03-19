# %%
import numpy as np
from IPython.display import display

from backpropy.gradient import Lin, Softmax, CE, Composition, Model, BatchGradientDescent

np.set_printoptions(suppress=True)

# %%
x2 = np.array([[1], [2], [-1]])
y2 = np.array([[2], [-3]])

X_train = x2.reshape(1, 3)
y_train = y2.reshape(1, 2)

z_1_prime = Lin(np.array([[0, 3], [-2, 1], [1, -1]]), np.array([[1], [-2]]))
f_1_prime = Softmax()

loss2 = CE()

# %%
z_1_prime(x2)

# %%

myoptimizer = BatchGradientDescent(5, 0.01, 1)
model2 = Model(
    [z_1_prime, f_1_prime],
    loss2,
    X_train,
    y_train,
    myoptimizer,
)
y_pred2 = model2.predict(x2)

# %%
print('prediction:')
print(y_pred2)
assert all(y_pred2 == f_1_prime(z_1_prime(x2)))
# %%
print('loss:')
print(model2.loss(y2, y_pred2))

# %%
print('weight vector:')
print(model2.composition.weight_vector)
# %%
assert all(np.isclose(-(y2 - y_pred2*y2.sum()), np.matmul(f_1_prime.input_gradient(z_1_prime(x2)), loss2.input_gradient(y2, y_pred2))))

# %%
display(model2.loss_x_y(x2, y2))
# %%

# %%
for i in range(100):
    model2.composition.weight_vector = model2.composition.weight_vector - 0.005 * model2.loss_params_gradient(x2, y2)
    # print(model2.composition.weight_vector)
    # print(0.0001 * model2.loss_params_gradient(x2, y2))
    print(model2.loss_x_y(x2, y2))
# %%
model2.composition.weight_vector = model2.composition.weight_vector - 0.005 * model2.loss_params_gradient(x2, y2)

# print(model2.composition.weight_vector)
# print(0.0001 * model2.loss_params_gradient(x2, y2))
print(model2.predict(x2))
print(model2.loss_x_y(x2, y2))
# %%
x2 = np.array([[1], [2], [-1]])
y2 = np.array([[2], [-3]])

X_train = x2.reshape(1, 3)
y_train = y2.reshape(1, 2)

W = np.array([[0, 3], [-2, 1], [1, -1]])
bias = np.array([[1], [-2]])
z_1 = Lin(W, bias)
f_1 = Softmax()
loss = CE()
myoptimizer = BatchGradientDescent(batch_size=5, learning_rate=0.01, epochs=3)

model = Model(
    [z_1, f_1],
    loss,
    X_train,
    y_train,
    myoptimizer,
)

model.fit()