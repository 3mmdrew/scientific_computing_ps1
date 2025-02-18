import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc
from tqdm import tqdm
import matplotlib.animation as animation# Initialize concentration matrix

### ANALYTIC SOLUTION FOR COMPARISON
def analytic_solution(x, t, D, num_terms):
    if t == 0:
        return 0  # Boundary condition
    solution = 0
    for i in range(num_terms):
        term1 = erfc((1 - x + 2 * i) / (2 * np.sqrt(D * t)))
        term2 = erfc((1 + x + 2 * i) / (2 * np.sqrt(D * t)))
        solution += term1 - term2
    return solution

def jacobi_iteration(c, N, epsilon = 1e-5, max_iter=10000):
    c_next = c.copy()
    results = {}
    for k in range(max_iter): 
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

        delta = np.max(np.abs(c_next-c))

        results[k] = {"c": c_next.copy()[0], "delta": delta}

        if delta < epsilon:
            print("J: ", k)
            return results
        else:
            c[:] = c_next[:]
    
def Gauss_Seide_iteration(c, N, epsilon = 1e-5, max_iter=10000):
    c_next = c.copy()
    results = {}

    for k in range(max_iter): 
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
                    * (c[1, j] + c_next[N - 1, j] + c[N, j + 1] + c_next[N, j - 1])
                )
            
        c_next[0, 1:N] = c_next[N, 1:N]

        # Apply fixed boundary conditions
        c_next[:, N] = 1.0  # Top boundary
        c_next[:, 0] = 0.0  # Bottom boundary

        delta = np.max(np.abs(c_next-c))

        results[k] = {"c": c_next.copy()[0], "delta": delta}

        if delta < epsilon:
            print("GS: ", k)
            return results
        else:
            c[:] = c_next[:]

def SOR_iteration(c, w, N, epsilon = 1e-5, max_iter=10000):
    c_next = c.copy()
    results = {}

    for k in range(max_iter): 
        for j in range(1, N):  # Skip boundaries in x-direction
            for i in range(1, N):  # Skip boundaries in y-direction
                c_next[i, j] = (
                    w * 0.25
                    * (c[i + 1, j] + c_next[i - 1, j] + c[i, j + 1] + c_next[i, j - 1])
                    + (1 - w) * c[i,j]
                )

        # Apply periodic boundary conditions in the x-direction
        for j in range (1, N):
            c_next[N,j] = (
                    w * 0.25
                    * (c[1, j] + c_next[N - 1, j] + c[N, j + 1] + c_next[N, j - 1])
                    + (1 - w) * c[N, j]
                )
            
        c_next[0, 1:N] = c_next[N, 1:N]

        # Apply fixed boundary conditions
        c_next[:, N] = 1.0  # Top boundary
        c_next[:, 0] = 0.0  # Bottom boundary

        delta = np.max(np.abs(c_next-c))

        results[k] = {"c": c_next.copy()[0], "delta": delta}

        if delta < epsilon:
            print("SOR: ", k)
            return results
        else:
            c[:] = c_next[:]

# Parameters
N = 50
D = 1
L = 1.0  # Domain
t = 0
delta_x = L / N

c0 = np.zeros((N + 1, N + 1))

# Boundary conditions (initial)
c0[:, N] = 1.0  # Top boundary
c0[:, 0] = 0.0  # Bottom boundary

results_J = jacobi_iteration(c0.copy(), N)
results_GS = Gauss_Seide_iteration(c0.copy(), N)
results_SOR = SOR_iteration(c0.copy(), 1.8, N)

''' 
### Colormap to show final concentrations
final_iteration_GS = max(results_GS.keys())
final_concentration_GS = results_GS[final_iteration_GS]["c"]

grid = np.zeros((N + 1, N + 1))
grid[:, :] = np.tile(final_concentration_GS, (N + 1, 1)).T  # Correct raster update

plt.imshow(grid,  origin='lower' )
plt.title(f"Concentration for G-S") 
plt.colorbar()
plt.show()
'''

''' 
### Caculation of optimal w
ws = np.arange(1.7, 1.9999, 0.01)
Ns = np.arange(15, 51, 5)

find_optimal_w = {}

for N in Ns:
    min_L = float('inf')  
    best_w = None 
    
    for w in ws:
        result = SOR_iteration(c, w, N)  # Call the function once

        if result is None:  # Correct check for None
            L = 0
        else:
            L = len(result)  # Get number of iterations

        # Update if a lower length is found
        if L < min_L:
            min_L = L
            best_w = w
    
    # Store the optimal w for this N
    find_optimal_w[N] = {"N": N, "w": best_w, "Length": min_L}

# Extract values
Ns_plot = list(find_optimal_w.keys())  # List of N values
ws_plot = [find_optimal_w[N]["w"] for N in Ns_plot]  # Corresponding optimal w values

plt.plot(Ns_plot, ws_plot)
plt.title("Optimal w value")
plt.xlabel("Size N")
plt.ylabel("Optimal w")
plt.show()
'''


'''
### Plot concentrations for y coordinate

# Extract concentration at last interation 
final_iteration_J = max(results_J.keys())
final_concentration_J = results_J[final_iteration_J]["c"]

final_iteration_GS = max(results_GS.keys())
final_concentration_GS = results_GS[final_iteration_GS]["c"]

final_iteration_SOR = max(results_SOR.keys())
final_concentration_SOR = results_SOR[final_iteration_SOR]["c"]

# Extract delta values for each iteration
iterations_J = list(results_J.keys())
deltas_J = [results_J[k]["delta"] for k in iterations_J]

iterations_GS = list(results_GS.keys())
deltas_GS = [results_GS[k]["delta"] for k in iterations_GS]

iterations_SOR = list(results_SOR.keys())
deltas_SOR = [results_SOR[k]["delta"] for k in iterations_SOR]

# Calculate the analytic solution
y = np.zeros(N+1)
for i in range(N+1):
    x = i/(N+1)
    y[i] = analytic_solution(x,100,D,100)

### Plot the concentration of each y-cordinate
plt.plot(range(N+1),final_concentration_J, alpha = 0.7, label = "Jacobi", color = 'blue')
plt.plot(range(N+1),final_concentration_GS, alpha = 0.7, label = "Gauss Seide", color = 'orange')
plt.plot(range(N+1),final_concentration_SOR, alpha = 0.7, label = "SOR", color = 'red')

plt.plot(range(N+1), y, label = "Analytic", linestyle='dashed', color = 'black')
plt.xlabel("y")
plt.ylabel("Concentration")
plt.legend()
plt.show()
'''

'''
### Plot the convergence at each iteration k

# Create a figure with three subplots (stacked vertically)
fig, axes = plt.subplots(3, 1, figsize=(8, 12)) # sharex=True  # Share x-axis for alignment

# **Plot 1: Jacobi Method**
axes[0].plot(iterations_J, deltas_J, alpha=0.7, label="Jacobi", color='blue')
axes[0].set_ylabel("Delta")
axes[0].set_yscale("log")
axes[0].legend()


# **Plot 2: Gauss-Seidel Method**
axes[1].plot(iterations_GS, deltas_GS, alpha=0.7, label="Gauss-Seidel", color='orange')
axes[1].set_ylabel("Delta")
axes[1].set_yscale("log")
axes[1].legend()


# **Plot 3: SOR Method**
axes[2].plot(iterations_SOR, deltas_SOR, alpha=0.7, label="SOR", color='red')
axes[2].set_xlabel("Iteration")
axes[2].set_ylabel("Delta")
axes[2].set_yscale("log")
axes[2].legend()


# Adjust layout for better spacing
plt.tight_layout()
plt.title("Convergence of Delta")
plt.show()

'''

