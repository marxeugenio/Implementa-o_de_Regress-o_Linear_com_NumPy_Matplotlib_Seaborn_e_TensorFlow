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

# Gráfico de Regressão Linear com TensorFlow (Azul)
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label="Dados Originais")
plt.plot(x, W * x + b, color="blue", label="Regressão Linear TensorFlow")  # Linha azul para a regressão do TensorFlow
plt.legend()
plt.title("Regressão Linear com TensorFlow (Azul)\n"
          "Este gráfico apresenta a linha de regressão gerada pelo modelo criado com TensorFlow. "
          "A linha mostra a melhor aproximação linear para os dados, buscando minimizar a diferença entre os pontos e a linha.")

# Gráfico de Regressão Linear com SciPy (Verde)
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label="Dados Originais")
plt.plot(x, intercept + slope * x, color="green", label="Regressão Linear SciPy")  # Linha verde para a regressão do SciPy
plt.legend()
plt.title("Regressão Linear com SciPy (Verde)\n"
          "Aqui, a linha de regressão é calculada usando SciPy. "
          "Comparada com a linha gerada pelo TensorFlow, esta linha demonstra uma aproximação similar, "
          "ilustrando a relação linear entre as variáveis.")

# Gráfico de Regressão Linear com Seaborn (Azul)
plt.figure(figsize=(8, 6))
sns.regplot(x=x, y=y, scatter=True, color="blue")
plt.title("Gráfico de Regressão Linear com Seaborn (Azul)\n"
          "Este gráfico é gerado utilizando a função de regressão linear do Seaborn. "
          "Ele mostra uma linha de tendência juntamente com uma nuvem de dispersão dos dados, "
          "oferecendo uma visão da relação linear entre as variáveis, similar aos gráficos anteriores.")

# Gráfico de Dispersão dos Dados Originais
plt.figure(figsize=(8, 6))
plt.scatter(x, y)
plt.title("Gráfico de Dispersão dos Dados Originais\n"
          "Este gráfico exibe os dados originais utilizados para a regressão linear. "
          "Os pontos representam a relação entre as variáveis independentes e dependentes, "
          "mostrando a distribuição dos dados.")

plt.show()
