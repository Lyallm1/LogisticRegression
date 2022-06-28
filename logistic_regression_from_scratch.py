import numpy as np, matplotlib.pyplot as plt

np.random.seed(12)
simulated_separable_features = np.vstack((np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], 5000), np.random.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], 5000))).astype(np.float32)
simulated_labels = np.hstack((np.zeros(5000), np.ones(5000)))
plt.figure(figsize=(12, 8))
plt.scatter(simulated_separable_features[:, 0], simulated_separable_features[:, 1], c=simulated_labels, alpha=0.4)
plt.show()

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))
def log_likelihodd(features, target, weights):
    scores = features.dot(weights)
    return np.sum(target * scores - np.log(1 + np.exp(scores)))
def logistic_regression(features, target, num_steps, lr, add_intercept=False):
    if add_intercept: features = np.hstack((np.ones((features.shape[0], 1)), features))
    weights = np.zeros(features.shape[1])
    for step in range(num_steps):
        scores = features.dot(weights)
        predictions = sigmoid(scores)
        