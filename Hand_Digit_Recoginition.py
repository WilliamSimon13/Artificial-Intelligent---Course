# import numpy as np
# from sklearn.datasets import fetch_openml
# import matplotlib.pyplot as plt

# x_all, y_all = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
# x0 = x_all[np.where(y_all == '0')[0]]
# x1 = x_all[np.where(y_all == '1')[0]]

# plt.imshow(x1[100].reshape(28, 28))
# plt.imshow(x0[1].reshape(28, 28))

# x0 = x0[:1000, :]
# x1 = x1[:1000, :]

# y0 = np.zeros((x0.shape[0]))
# y1 = np.ones((x1.shape[0]))

# X = np.concatenate((x0, x1), axis=0)
# y = np.concatenate((y0, y1), axis=0)

# one = np.ones((X.shape[0], 1))
# X = np.concatenate((one, X), axis=1)    

# def sigmoid(s):
#     return 1 / (1 + np.exp(-s))
# def sgd(X, y, theta_init, eta = 0.05, gamma = 0.9):
#     theta_old = theta_init
#     theta_epoch = theta_init
#     v_old = np.zeros_like(theta_init.shape)
#     N= X.shape[0]
#     for it in range(100):  # max epochs
#         mix_id = np.random.permutation(N)  # shuffle dataset
#         for i in mix_id:  # loop each sample
#             xi = X[i, :]   # (1,785)
#             yi = y[i]      # scalar
#             hi = sigmoid(np.dot(xi, theta_old))  # scalar
#             gi = xi * (hi - yi)                 # (785,1)

#             v_new = gamma * v_old + eta * gi

#             theta_new = theta_old - v_new

#             theta_old = theta_new
             
#             v_old = v_new

#         # convergence check (per epoch)
#         if np.linalg.norm(theta_epoch - theta_old) < 1e-3:
#             break

#         theta_epoch = theta_old

#     return (theta_epoch, it)

# theta_init = np.random.rand(1,X.shape[1])[0]
# (theta, it) = sgd(X, y, theta_init)
# print('Theta = ', theta, 'Iteration = ', it)

# np.savetxt('theta.txt', theta)
# import numpy as np
# theta = np.loadtxt('theta.txt')


import numpy as np
from sklearn.datasets import fetch_openml

# Lấy dữ liệu MNIST chỉ với số 0 và 1
x_all, y_all = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
x0 = x_all[y_all == "0"][:1000]
x1 = x_all[y_all == "1"][:1000]

y0 = np.zeros(x0.shape[0])
y1 = np.ones(x1.shape[0])

X = np.concatenate((x0, x1), axis=0)
y = np.concatenate((y0, y1), axis=0)

# Chuẩn hóa dữ liệu về [0,1]
X = X / 255.0

# Thêm bias (cột 1)
one = np.ones((X.shape[0], 1))
X = np.concatenate((one, X), axis=1)  # (2000, 785)

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def sgd(X, y, theta_init, eta=0.05, gamma=0.9, max_epochs=100):
    theta_old = theta_init.copy()
    theta_epoch = theta_init.copy()
    v_old = np.zeros_like(theta_init)
    N = X.shape[0]

    for it in range(max_epochs):
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[i, :].reshape(-1, 1)  # (785,1)
            yi = y[i]

            hi = sigmoid(np.dot(theta_old.T, xi))  # scalar
            gi = xi * (hi - yi)                    # (785,1)

            v_new = gamma * v_old + eta * gi
            theta_new = theta_old - v_new

            theta_old = theta_new
            v_old = v_new

        # check convergence
        if np.linalg.norm(theta_epoch - theta_old) < 1e-3:
            break
        theta_epoch = theta_old.copy()

    return theta_old, it

# Khởi tạo theta (785,1)
theta_init = np.random.rand(X.shape[1], 1)
theta, it = sgd(X, y, theta_init)

print("Training done after", it, "epochs")
np.savetxt("theta.txt", theta)

# --- Test hàm dự đoán ---
def predict(x, theta, low=0.2, high=0.8):
    """
    Trả về 0 nếu chắc chắn là số 0
    Trả về 1 nếu chắc chắn là số 1
    Trả về -1 nếu không chắc (không phải 0/1)
    """
    x = x / 255.0
    one = np.ones((1, 1))
    x = np.concatenate((one, x.reshape(1, -1)), axis=1)  # (1,785)
    prob = sigmoid(np.dot(x, theta))
    if prob >= high:
        return 1
    elif prob <= low:
        return 0
    else:
        return -1  # Không chắc chắn
