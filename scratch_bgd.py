import numpy as np

# Data
x = np.random.rand(100, 1)
y = 32 + 1.8*x + .01*np.random.randn(100, 1)
N = x.shape[0]

# Add bias column
one = np.ones((x.shape[0], 1))
X = np.concatenate((one, x), axis=1)

# Gradient function
def gradient(theta):
    return 1/N * X.T.dot(X.dot(theta) - y)

# Batch Gradient Descent
def batch_gradient_descent(theta_init, learning_rate):
    theta = theta_init
    for it in range(100):
        theta_new = theta - learning_rate * gradient(theta)
        if np.linalg.norm(gradient(theta_new))/len(theta_new) < 1e-3:
            break
        theta = theta_new
    return (theta, it)

(theta, iteration) = batch_gradient_descent(theta_init=np.array([[5], [5]]),
                                            learning_rate=1)
print('Theta = ', theta.T)
print('Iteration = ', (iteration+1))