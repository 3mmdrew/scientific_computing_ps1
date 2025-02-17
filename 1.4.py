import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc
from tqdm import tqdm
import matplotlib.animation as animation# Initialize concentration matrix

# Parameters
N = 50
D = 1
L = 1.0  # Domain
t = 0
delta_x = L / N

c = np.zeros((N + 1, N + 1))

# Boundary conditions (initial)
c[:, N] = 1.0  # Top boundary
c[:, 0] = 0.0  # Bottom boundary


def jacobi_iteration(c, epsilon = 1e-5):
    c_next = c.copy()
    i = 1
    while i > 0: 
        for i in range(1, N):  # Skip boundaries in x-direction
            for j in range(1, N):  # Skip boundaries in y-direction
                c_next[i, j] = (
                    0.25
                    * (c[i + 1, j] + c[i - 1, j] + c[i, j + 1] + c[i, j - 1])
                )

        # Apply periodic boundary conditions in the x-direction
        for j in range (1,N):
            c_next[N,j] = (
                    0.25
                    * (c[1, j] + c[N - 1, j] + c[N, j + 1] + c[N, j - 1])
                )
            
        c_next[0, 1:N] = c_next[N, 1:N]

        # Apply fixed boundary conditions
        c_next[:, N] = 1.0  # Top boundary
        c_next[:, 0] = 0.0  # Bottom boundary

        if np.max(np.abs(c_next-c)) < epsilon:
            return c_next
        else:
            c[:] = c_next[:]
    
def  Gauss_Seide_iteration(c, epsilon = 1e-5):
    c_next = c.copy()
    i = 1
    while i>0: 
        for j in range(1, N):  # Skip boundaries in x-direction
            for i in range(1, N):  # Skip boundaries in y-direction
                c_next[i, j] = (
                    0.25
                    * (c[i + 1, j] + c_next[i - 1, j] + c[i, j + 1] + c_next[i, j - 1])
                )

        # Apply periodic boundary conditions in the x-direction
        for j in range (1,N):
            c_next[N,j] = (
                    0.25
                    * (c[1, j] + c[N - 1, j] + c[N, j + 1] + c[N, j - 1])
                )
            
        c_next[0, 1:N] = c_next[N, 1:N]

        # Apply fixed boundary conditions
        c_next[:, N] = 1.0  # Top boundary
        c_next[:, 0] = 0.0  # Bottom boundary

        if np.max(np.abs(c_next-c)) < epsilon:
                return c_next
        else:
            c[:] = c_next[:]

first = jacobi_iteration(c)
second = Gauss_Seide_iteration(c)

plt.imshow(first.T,  origin='lower' )
plt.title(f"Concentration using Jacobi Iteration") 
plt.colorbar()
plt.show()

# print(first)
# print(second)

