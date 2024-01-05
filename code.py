import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from scipy.stats import linregress

np.random.seed(0)
x = np.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.randn(100) * 2  

slope, intercept, r_value, p_value, std_err = linregress(x, y)

X = tf.constant(x, dtype=tf.float32)  
Y = tf.constant(y, dtype=tf.float32)

W = tf.Variable(np.random.randn(), name="weight")  
b = tf.Variable(np.random.randn(), name="bias")  

def linear_regression(x):
    return W * x + b

def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

learning_rate = 0.01
optimizer = tf.optimizers.SGD(learning_rate)

epochs = 1000
for epoch in range(epochs):
    with tf.GradientTape() as tape:  
        predictions = linear_regression(X)
        loss = mean_squared_error(Y, predictions)

    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))

plt.figure(figsize=(8, 6))
plt.scatter(x, y, label="Dados Originais")
plt.plot(x, W * x + b, color="red", label="Regressão Linear TensorFlow")
plt.plot(x, intercept + slope * x, color="green", label="Regressão Linear SciPy")
plt.legend()

plt.figure(figsize=(8, 6))
sns.regplot(x=x, y=y, scatter=True, color="blue")
plt.title("Regressão Linear com Seaborn")

plt.show()
plt.figure(figsize=(8, 6))
sns.regplot(x=x, y=y, scatter=True, color="blue")
plt.title("Regressão Linear com Seaborn")

plt.show()
