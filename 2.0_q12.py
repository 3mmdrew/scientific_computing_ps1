import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc
from tqdm import tqdm
import matplotlib.animation as animation

# Parameters
N = 50
D = 1
L = 1.0  # Domain
delta_x = L / N

max_delta_t = delta_x**2 / (4 * D)
delta_t = 0.0001
assert delta_t <= max_delta_t, f"delta_t must be <= {max_delta_t}, right now it is {delta_t}"

max_time = 1.0 + max_delta_t  # Simulation time
simulation_steps = int(max_time / delta_t)  # Number of time steps

# Initialize concentration array
c = np.zeros(N + 1)

# Boundary conditions (initial)
c[N] = 1.0  # Top boundary
c[0] = 0.0  # Bottom boundary

### TIME-STEPPING SIMULATION
def upd_concentration(c, D, delta_t, delta_x):
    c_next = c.copy()
    for j in range(1, N):  # Skip boundaries
        c_next[j] = (
            c[j]
            + (delta_t * D / delta_x**2)
            * (c[j + 1] + c[j - 1] - 2 * c[j])
        )

    # Apply fixed boundary conditions
    c_next[N] = 1.0  # Top boundary
    c_next[0] = 0.0  # Bottom boundary

    return c_next


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


### SIMULATION

x_values = np.linspace(0, 1, N + 1)
y_values = np.linspace(0, 1, N + 1)

stepping_history = {}
analysis_history = {}


for t in tqdm(range(simulation_steps), desc="Diffusing:"):
    c = upd_concentration(c, D, delta_t, delta_x)

    
    if t*delta_t in [0, 0.001, 0.01, 0.1, 1.0]:
        stepping_history[t*delta_t] = c
    
    if t*delta_t in [0, 1.0]:
        analysis_history[t*delta_t] = [analytic_solution(x, t, D, simulation_steps) for x in x_values] 
     

# grid = np.zeros((N + 1, N + 1))
# for i in range(0,N+1):
#     for j in range(0,N+1):
#         grid[i,j] = c[i]

# plt.imshow( grid,  origin='lower' )
# plt.title(f"Concentration at t = {simulation_steps*delta_t}") 
# plt.colorbar()
# plt.show()


# Animatie instellen
fig, ax = plt.subplots()
grid = np.zeros((N + 1, N + 1))
im = ax.imshow(np.zeros((N + 1, N + 1)), origin='lower', vmin=0, vmax=1)
cbar = plt.colorbar(im)
ax.set_title("Concentration over time")

# Only show frames at t =  0.1, 0.2, 0.3, etc.
# valid_frames = [int(t / delta_t) for t in np.arange(0.1, simulation_steps * delta_t, 0.1)]

def update(frame):
    global c
    #steps_to_run = valid_frames[frame] - (valid_frames[frame - 1] if frame > 0 else 0)
    c = upd_concentration(c, D, delta_t, delta_x)
    grid[:, :] = np.tile(c, (N + 1, 1)).T  # Correct raster update
    im.set_array(grid)
    ax.set_title(f"Concentration at t = ") #{t:.2f}
    return [im]

ani = animation.FuncAnimation(fig, update, frames=simulation_steps, interval=50)
plt.show()


# # Plot the concentration profile
# for i,c in enumerate(stepping_history.values()):
#     concentration_profile = c[int(N / 2), :]  # Concentration at the middle of the x-domain
#     plt.plot(y_values, concentration_profile, label=f"(t = {list(stepping_history.keys())[i]})")
# for i,a in enumerate(analysis_history.values()):
#     plt.plot(x_values, a, label=f'Analytic (t={list(analysis_history.keys())[i]})')

# """  
# analytic_vals = [analytic_solution(x, t, D, simulation_steps) for x in x_values]    

# """

# plt.xlabel('y')
# plt.ylabel('Concentration c(y,t)')
# plt.title('Concentration Profile at Different Times')
# plt.grid(visible=True)
# plt.legend()
# plt.show()

