import matplotlib.pyplot as plt
import numpy as np

# Generate two random clusters of 2D data
N_c = 100
A = 0.3*np.random.randn(N_c, 2)+[1, 1]
B = 0.3*np.random.randn(N_c, 2)+[3, 3]
X = np.hstack((np.ones(2*N_c).reshape(2*N_c, 1), np.vstack((A, B))))
Y = np.vstack(((-1*np.ones(N_c)).reshape(N_c, 1), np.ones(N_c).reshape(N_c, 1)))
N = 2*N_c

# Run perceptron
delta = 1E-7
eta = 5E-1
max_iter = 500
w = np.array([0, 0, 0])
w_old = np.array([0, 0, 0])
for t in range(0, max_iter):
    for i in range(0, N):
        x_i = X[i, :]
        y_i = Y[i]
        h = np.sign(np.dot(w, x_i))
        if h != y_i:
            w = w+eta*y_i*x_i

    if 1/(float(N))*np.abs(np.sum(w_old-w)) < delta:
        print "Converged in", t, "steps."
        break

    w_old = w

    if t==max_iter-1:
        print "Warning, did not converge."

print "Weights found:",w


tt = np.linspace(np.min(X[:, 1])-1, np.max(X[:, 1])+1, 10)
bf_line = -w[0]/w[2]-w[1]/w[2]*tt

plt.plot(X[0:N_c-1, 1], X[0:N_c-1, 2], 'ro', X[N_c:, 1], X[N_c:, 2], 'bo', tt, bf_line, 'k-')
if -w[1]/w[2]<0:
    plt.fill_between(tt, -1, bf_line, facecolor='red', alpha=0.5)
    plt.fill_between(tt, bf_line, 5, facecolor='blue', alpha=0.5)
else:
    plt.fill_between(tt, -1, bf_line, facecolor='blue', alpha=0.5)
    plt.fill_between(tt, bf_line, 5, facecolor='red', alpha=0.5)

plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.axis([np.min(X[:, 1])-0.5, np.max(X[:, 1])+0.5, np.min(X[:, 2])-0.5, np.max(X[:, 2])+0.5])
plt.savefig('two_class.png')
plt.show()

