import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import dft


# For Matrices with no complex values 
def createS(t, vecs):
    S = np.zeros(128)
    i = 0
    for v in vecs:
        if v > t:
            S[i] = v - t
        elif v < -t:
            S[i] = v + t
        i += 1
    return S


def createScomplex(t, vecs):
    S = np.zeros(128, dtype='complex_')
    i = 0
    for v in vecs:
        R = np.maximum(np.abs(v) - t, 0)
        angle = np.angle(v)
        S[i] = np.exp(1j*angle)*R
        i += 1
    return S


def lassoADMM(A, b, lamda, iteration, rho, true_x):

    z = np.random.normal(0, 0.01, 128)
    u = np.random.normal(0, 0.01, 128)
    x = np.random.normal(0, 0.01, 128)
    error = []
    error.append(np.linalg.norm(true_x - x) / np.sqrt(128))

    for _ in range(iteration):

        x = np.linalg.inv(A.conj().T@A + rho*np.identity(128)) @ (np.dot(A.conj().T, b) + rho * (z - u))
        z = createScomplex(lamda/rho, (x + u))
        u = u + x - z
        error.append(np.linalg.norm(true_x-x) / np.sqrt(128))
    return x, z, error


if __name__ == '__main__':

    #  Todo: Part 1
    x = np.zeros(128)
    index = np.random.randint(0, 128, 5)
    randoms = [0.2, 0.4, 0.6, 0.8, 1]

    for ind, ran in zip(index, randoms):
        x[ind] = ran

    y = np.random.normal(0, 0.05, 128) + x
    lamdas = [0.01, 0.05, 0.1, 0.2]
    errors1 = []
    errors2 = []

    for lamda in lamdas:
        x_pred = np.zeros(128)
        for i in range(128):
            if y[i] > lamda:
                x_pred[i] = y[i] - lamda
            if y[i] < -lamda:
                x_pred[i] = y[i] + lamda
        errors1.append(np.linalg.norm(x_pred - x))
        errors2.append(np.linalg.norm(x_pred - y))

    plt.plot(lamdas, errors1, label='L2 Norm of ||x^ - x||', color='b', marker='o')
    # plt.plot(lamdas, errors2, label='L2 Norm of ||x^ - y||', color='orange', marker='o')
    plt.xlabel("Lambda Values")
    plt.ylabel("L2 Norms")
    plt.legend()
    plt.show()
    #  Todo: Part 1

    #  Todo: Part 2
    x = np.zeros(128)
    index = np.random.randint(0, 128, 5)
    randoms = [0.2, 0.4, 0.6, 0.8, 1]
    for ind, ran in zip(index, randoms):
        x[ind] = ran

    F = dft(128)
    Fc = np.roll(F, 64)
    M = np.zeros((128, 128))
    for i in range(0, 128, 4):
        M[i][i] = 1
    Fu = M @ F

    X = Fc @ x
    Fu_inv = np.linalg.pinv(Fu)

    Xu = np.zeros(128, dtype='complex_')
    for i in range(0, 128, 4):
        Xu[i] = X[i]

    xu = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Xu)))*4
    mnls = np.dot(Fu_inv, X)

    fig, axs = plt.subplots(2)
    fig.suptitle('Proof that xu is the MNLS solution')
    axs[0].plot(np.abs(mnls), label='MNLS solution')
    axs[0].legend()
    axs[1].plot(np.abs(xu), label='xu', color='orange')
    axs[1].legend()
    for ax in axs.flat:
        ax.label_outer()
    plt.show()

    Xr = np.zeros(128, dtype='complex_')
    indexs = np.random.randint(0, 128, 32)
    for i in indexs:
        Xr[i] = X[i]

    xr = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Xr)))*4

    fig, axs = plt.subplots(4)
    fig.suptitle('Indexes & Absolute Values')
    axs[0].plot(x, label='x')
    axs[0].legend()
    axs[1].plot(np.abs(xu), label='xu', color='orange')
    axs[1].legend()
    axs[2].plot(np.abs(xr), label='xr', color='green')
    axs[2].legend()
    axs[3].plot(x, label='x')
    axs[3].plot(np.abs(xu), label='xu', color='orange')
    axs[3].plot(np.abs(xr), label='xr', color='green')
    for ax in axs.flat:
        ax.label_outer()
    plt.show()
    #  Todo: Part 2

    #  Todo: Part 3
    F = dft(128)
    F = np.roll(F, 64)
    M = np.zeros((128, 128))
    index = np.random.randint(0, 128, 32)
    for i in index:
        M[i][i] = 1
    Fu = M @ F

    x_true = np.zeros(128)
    index = np.random.randint(0, 128, 5)
    randoms = [0.2, 0.4, 0.6, 0.8, 1]

    for ind, ran in zip(index, randoms):
        x_true[ind] = ran

    y = np.random.normal(0, 0.05, 128) + x_true
    y = np.dot(F, y)
    # lamdas = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    lamdas = [0.01, 0.05, 0.1]
    iteration = 5000
    rho = 10
    # rho = 100

    xs = []
    zs = []

    for lamda in lamdas:
        x, z, error = lassoADMM(Fu, y, lamda, iteration, rho, x_true)
        # x, z, error = lassoADMM(Fu, y, 1, iteration, rho, x_true)
        xs.append(x)
        zs.append(z)
        plt.plot(error, label='Lambda = ' + str(lamda))
    plt.xlabel("Iteration Number")
    plt.ylabel("MSE of (x - x^)")
    plt.legend()
    plt.show()

    fig, axs = plt.subplots(5)
    fig.suptitle('Indexes & Absolute Values')
    axs[0].plot(x_true, label='x')
    axs[0].legend()

    axs[1].plot(np.abs(xs[0]), label='Lambda = ' + str(lamdas[0]), color='orange')
    axs[1].legend()

    axs[2].plot(np.abs(xs[1]), label='Lambda = ' + str(lamdas[1]), color='green')
    axs[2].legend()

    axs[3].plot(np.abs(xs[2]), label='Lambda = ' + str(lamdas[2]), color='red')
    axs[3].legend()

    axs[4].plot(x_true, label='x')
    axs[4].plot(np.abs(xs[0]), label='Lambda = ' + str(lamdas[0]), color='orange')
    axs[4].plot(np.abs(xs[1]), label='Lambda = ' + str(lamdas[1]), color='green')
    axs[4].plot(np.abs(xs[2]), label='Lambda = ' + str(lamdas[2]), color='red')
    for ax in axs.flat:
        ax.label_outer()
    plt.show()
    #  Todo: Part 3