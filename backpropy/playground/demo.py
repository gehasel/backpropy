import numpy as np
import logging

import backpropy.mnist_for_numpy.mnist as mnist
from backpropy.gradient import Model, BatchGradientDescent, CE, Lin, Softmax, ReLU, accuracy

logging.basicConfig(level=logging.INFO)


def demo(
    num_training_datapoints=500,
    batch_size=10,
    learning_rate=0.05,
    epochs=5,
):
    X_train, y_train, X_test, y_test = mnist.load()
    num_training_datapoints = 500
    X_train = X_train[:num_training_datapoints] / 255.
    X_test = X_test / 255.

    def one_hot_encode_number_labels(arr):
        one_hot = np.zeros((arr.size, 10))
        one_hot[np.arange(arr.size), arr] = 1
        return one_hot

    y_train_ohe = one_hot_encode_number_labels(y_train[:num_training_datapoints])

    rng = np.random.default_rng(seed=123)

    lower_rand = -0.1
    upper_rand = 0.1
    in1 = 784
    out1 = 300
    z_1 = Lin(
        rng.uniform(low=lower_rand, high=upper_rand, size=(in1, out1)),
        rng.uniform(low=lower_rand, high=upper_rand, size=(out1, 1)),
        name="z_1",
    )
    f_1 = ReLU(name="f_1")

    out2 = 100
    z_2 = Lin(
        rng.uniform(low=lower_rand, high=upper_rand, size=(out1, out2)),
        rng.uniform(low=lower_rand, high=upper_rand, size=(out2, 1)),
        name="z_2",
    )
    f_2 = ReLU(name="f_2")

    out3 = 10
    z_3 = Lin(
        rng.uniform(low=lower_rand, high=upper_rand, size=(out2, out3)),
        rng.uniform(low=lower_rand, high=upper_rand, size=(out3, 1)),
        name="z_3",
    )
    f_3 = Softmax(name="f_3")

    loss = CE()

    myoptimizer = BatchGradientDescent(batch_size=batch_size, learning_rate=learning_rate, epochs=epochs)
    mymodel = Model(
        [z_1, f_1, z_2, f_2, z_3, f_3],
        loss,
        X_train,
        y_train_ohe,
        myoptimizer,
    )
    mymodel.fit()
    y_pred = np.array([np.argmax(mymodel.predict(x)) for x in X_test])
    myaccuracy = accuracy(y_test, y_pred)
    print(f"Achieved Accuracy: {myaccuracy}")


if __name__ == "__main__":
    num_training_datapoints = 500
    batch_size=10
    learning_rate=0.05
    epochs=5
    demo(num_training_datapoints, batch_size, learning_rate, epochs)
