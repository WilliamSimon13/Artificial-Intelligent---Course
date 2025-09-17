import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100, 1)
Y = 32 + 1.8*X + .01*np.random.randn(100, 1)

N = X.shape[0]

one = np.ones((X.shape[0], 1))
X = np.concatenate((one, X), axis=1)

def stochastic_gradient_descent(theta_init, learning_rate):
    theta = theta_init
    theta_epoch = theta_init

    for it in range(100):  # max epochs
        random_id = np.random.permutation(N)  # shuffle dataset
        for i in random_id:  # loop each sample
            xi = X[i, :]   # (1,2)
            yi = Y[i]      # scalar
            gi = (xi.T * (np.dot(xi, theta) - yi)).reshape(2,1)

            theta_new = theta - learning_rate * gi
            theta = theta_new

        # convergence check (per epoch)
        if np.linalg.norm(theta_epoch - theta)/len(theta_init) < 1e-3:
            return (theta_epoch, it)

        theta_epoch = theta

    return (theta_epoch, it)


(theta, iteration) = stochastic_gradient_descent(theta_init=np.array([[5], [5]]),
                                               learning_rate=0.1)
print('Theta = ', theta.T)
print('Iteration = ', (iteration+1))

# ---- Plotting result ----
plt.scatter(X[:,1], Y, color="blue", label="Data")   # scatter plot of real data

# predicted line
x_line = np.linspace(0, 1, 100).reshape(-1, 1)
x_line_bias = np.c_[np.ones((100,1)), x_line]       # add bias
y_pred = x_line_bias.dot(theta)

plt.plot(x_line, y_pred, color="red", linewidth=2, label="SGD Regression Line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Stochastic Gradient Descent Linear Regression")
plt.show()