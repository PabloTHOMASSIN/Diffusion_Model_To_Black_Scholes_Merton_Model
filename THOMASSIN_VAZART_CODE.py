# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Définition des variables Black-Scholes
sigma = 0.5  # volatilité du prix de l'action
E = 95   # prix d'exercice de l'option
T = 3/12   # temps avant échéance
r = 0.01   # taux d'intérêt
S = 100   # prix actuel de l'actif sous-jacent
N = 100  # resolution de la simulation


"""
    Différences finies
"""

def black_scholes_matrix(S, T, r, E, sigma, N):

    # Variables
    s = np.linspace(0.01, S+S*0.1, N)
    t = np.linspace(0.01, T+T*0.1, N)

    # Changement des variables
    k = 2*r / sigma**2
    alpha1 = (1-k) / 2
    beta = alpha1**2 + (k-1)*alpha1 - k
    tau_max = T * sigma**2 / 2

    # Espace dual
    x = np.log(s/E)
    tau = 2*sigma**2 * (T-t)
    
    # Différentielles
    dtau = tau_max/N
    dx = (x[-1] - x[0])/N
    alpha = dtau / dx**2

    u = np.zeros((x.size, tau.size))

    # Création de la matrice C
    C = np.diag(np.ones(N)*alpha + 1) + np.diag(np.ones(N-1)*(-alpha/2), 1) + np.diag(np.ones(N-1)*(-alpha/2), -1)

    # Inversion de la matrice C 
    C_inv = np.linalg.inv(C)

    # Initialisation
    u[:,0] = np.maximum(np.exp((k+1)*x/2) - np.exp((k-1)*x/2), 0)

    # Calcul de u(x, tau)
    for m in range(1, N):
        u[0, m] = 0
        u[-1, m] = (S/E) * np.exp(alpha*x[-1] + beta*tau[m])
        b = np.zeros(N)
        for n in range (1, N-1):
            b[n] = (1-alpha)*u[n,m-1] + (1/2)*(u[n-1,m-1]+u[n+1,m-1])
        u[:,m] = C_inv @ b.T
        
    s = E * np.exp(x)
    t = T - tau / (2*sigma**2)
    Call = E * u @ np.exp(-alpha * x[:, np.newaxis] - beta * tau)
    
    S, t = np.meshgrid(np.log(S), t)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(t, s, np.log(E * np.exp(-alpha * x[:, np.newaxis] - beta * tau) @ u), cmap='coolwarm')
    ax.set_xlabel('temps à maturité')
    ax.set_ylabel("prix de l'action")
    ax.set_zlabel("prix du call")
    ax.set_title('C en fonction de S et t')
    plt.show()

    return Call, s, t, alpha, beta


def black_scholes_value(S, T, r, E, sigma, N):
    # Variables
    s = np.linspace(0.01, S+S*0.1, N)
    t = np.linspace(0.01, T+T*0.1, N)
    
    # Changement des variables
    k = 2*r / sigma**2
    alpha1 = (1-k) / 2
    beta = alpha1**2 + (k-1)*alpha1 - k
    tau_max = T * sigma**2 / 2

    # Espace dual
    x = np.log(s/E)
    tau = 2*sigma**2 * (T-t)
    
    # Différentielles
    dtau = tau_max/N
    dx = (x[-1] - x[0])/N
    alpha = dtau / dx**2

    u = np.zeros((x.size, tau.size))

    # Création de la matrice C
    C = np.diag(np.ones(N)*alpha + 1) + np.diag(np.ones(N-1)*(-alpha/2), 1) + np.diag(np.ones(N-1)*(-alpha/2), -1)

    # Inversion de la matrice C 
    C_inv = np.linalg.inv(C)

    # Initialisation
    u[:,0] = np.maximum(np.exp((k+1)*x/2) - np.exp((k-1)*x/2), 0)

    # Calcul de u(x, tau)
    for m in range(1, N):
        u[0, m] = 0
        u[-1, m] = (S/E) * np.exp(alpha*x[-1] + beta*tau[m])
        b = np.zeros(N)
        for n in range (1, N-1):
            b[n] = (1-alpha)*u[n,m-1] + (1/2)*(u[n-1,m-1]+u[n+1,m-1])
        u[:,m] = C_inv @ b.T
        
    s = E * np.exp(x)
    t = T - tau / (2*sigma**2)
    Call = E * np.exp(-alpha * x[:, np.newaxis] - beta * tau) @ u
    	
    print("...")
    print(Call[-1, -1])
    
    return Call[-2, -2]

(call, s, t, alpha, beta) = black_scholes_matrix(S, T, r, E, sigma, N)


""" Affichage en fonction de la volatilité """

_sigma = np.linspace(0.0001, 1, N)
e = np.linspace(E-10, E+10, N)

Z=[]
for sigma  in _sigma:
    Z_row = []
    for E in e:
        Z_row.append(black_scholes_value(S, T, r, E, sigma, N))
    Z.append(Z_row)

X, Y = np.meshgrid(_sigma, e)
Z = np.array(Z).T

print(X)
print(Y)
print(Z)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, np.log(Z), cmap='coolwarm')
ax.set_xlabel('volatilité (sigma)')
ax.set_ylabel("prix d'excercice de l'action (E)")
ax.set_zlabel("prix du call")
ax.set_title('Prix du call en fonction de sigma et E')
ax.view_init(30, 45)
plt.show()




