import numpy as np
from scipy.special import expit, softmax

x = np.array([5, 7, 4, 3, 2])
y = np.array([0, 0, 1])
v1 = np.array([0.5, -0.1, -0.2, 0.1, -0.4])
v2 = np.array([-0.9, 0.7, 0.7, -0.5, 0.2])
w1 = np.array([0.1, 0.6])
w2 = np.array([0.8, 0.4])
w3 = np.array([0.7, -0.2])
b_v1, b_v2 = 0.02, -0.01
b_w1, b_w2, b_w3 = 0.0, 0.05, 0.04

# Hidden layer ReLU
v1_out = np.maximum(0, np.dot(v1, x) + b_v1)
v2_out = np.maximum(0, np.dot(v2, x) + b_v2)

# Output layer Sigmoid
w1_out = expit(np.dot(w1, [v1_out, v2_out]) + b_w1)
w2_out = expit(np.dot(w2, [v1_out, v2_out]) + b_w2)
w3_out = expit(np.dot(w3, [v1_out, v2_out]) + b_w3)

# Softmax
probs = softmax([w1_out, w2_out, w3_out])

# Cross-entropy loss
loss = -np.sum(y * np.log(probs))

print(f"Cross-entropy loss: {loss}")
