from src.NeuralNetwork import NeuralNetwork
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Load and split data
X, t = load_breast_cancer(return_X_y=True)
seed = 1

X_train, X_test, t_train, t_test = train_test_split(X, t, random_state=seed, train_size=0.75)

### Setup, train, and use network
NN = NeuralNetwork(input_nodes=X_train.shape[-1])
NN.addLayer(n_nodes=100, seed=seed,   act="Sigmoid") # Hidden layer
NN.addLayer(n_nodes=100, seed=seed+1, act="Sigmoid") # Hidden layer
NN.addLayer(n_nodes=  1, seed=seed+2, act=   "tanh") # Output layer
NN.set_LearningRate(eta=1e-4, method="Adam")
NN.set_CostFunc("CrossEntropy")
NN.train(X_train, t_train, epochs=10, seed=seed)
y = NN.classify(X_test)

### Plot results
conf = np.zeros((2, 2), dtype=float)
conf[0, 0] = sum((y == 0) & (t_test == 0))/(len(np.where(t_test == 0)[0])) # True Negative
conf[0, 1] = sum((y == 1) & (t_test == 0))/(len(np.where(t_test == 0)[0])) # False Positive
conf[1, 0] = sum((y == 0) & (t_test == 1))/(len(np.where(t_test == 1)[0])) # False Negative
conf[1, 1] = sum((y == 1) & (t_test == 1))/(len(np.where(t_test == 1)[0])) # True Positive
conf = np.round(conf, 2)

plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(conf, annot=True, ax=ax, cmap="viridis", cbar=False)
ax.set_title("Classification confusion matrix")
ax.set_ylabel("True value")
ax.set_xlabel("Network output")
plt.tight_layout()
plt.savefig("figures/classification.pdf", format="pdf")
plt.show()

acc = sum(y == t_test) / len(t_test)
print("Total accuracy:", acc)
