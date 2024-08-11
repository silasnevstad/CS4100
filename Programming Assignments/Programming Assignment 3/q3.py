import numpy as np

def calculate_gradients(x, y, v1, v2, w1, w2, w3):
    # Forward pass
    h = np.array([np.maximum(0, np.dot(v, x)) for v in [v1, v2]])
    z = np.array([1 / (1 + np.exp(-(np.dot(w, h)))) for w in [w1, w2, w3]])
    s = np.exp(z) / np.sum(np.exp(z))

    # Backward pass
    dL_dz = s - y

    # Calculate gradients for w1, w2, w3
    dL_dw = [dL_dz[i] * h for i in range(3)]

    # Calculate gradient for hidden layer
    dL_dh = np.dot(np.array([w1, w2, w3]).T, dL_dz)

    # Calculate gradients for v1, v2
    dL_dv = [dL_dh[i] * (np.dot(v, x) > 0) * x for i, v in enumerate([v1, v2])]

    return *dL_dw, *dL_dv

x = np.array([5, 7, 4, 3, 2])
y = np.array([0, 0, 1])
v1 = np.array([0.5, -0.1, -0.2, 0.1, -0.4])
v2 = np.array([-0.9, 0.7, 0.7, -0.5, 0.2])
w1 = np.array([0.1, 0.6])
w2 = np.array([0.8, 0.4])
w3 = np.array([0.7, -0.2])

gradients = calculate_gradients(x, y, v1, v2, w1, w2, w3)
for i, grad in enumerate(gradients):
    print(f"Gradient for {'w' if i < 3 else 'v'}{i % 3 + 1}:\n{grad}")