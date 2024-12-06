import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#define differential equation of SEIR model
def seir_eq(v, t, alpha, beta, gamma, N):
    S, E, I, R = v
    dS = -beta * S * I / N           # dS/dt = -βI/N*S
    dE = beta * S * I / N - alpha * E  # dE/dt = βI/N*S - αE
    dI = alpha * E - gamma * I        # dI/dt = αE - γI
    dR = gamma * I                    # dR/dt = γI 
    return [dS, dE, dI, dR]

# Initial state
ini_state = [3000, 0, 5, 0]  # [S0, E0, I0, R0]
t_max = 100
dt = 1
t = np.arange(0, t_max, dt)

# Total population
N = ini_state[0] + ini_state[1] + ini_state[2] + ini_state[3]

# Parameters
lp = 14  # latency period
ip = 7   # infectious period
R0 = 1.0  # Basic reproduction number
D = R0 * ip
alpha = 1 / lp
gamma = 1 / ip
R0 = 10.0  # Reproduction number changed
beta = R0 * gamma  # beta = R0 * gamma

# Print R0
print('R0 =', R0)

# Solve the differential equations
solution = odeint(seir_eq, ini_state, t, args=(alpha, beta, gamma, N))

# Extract the solutions for S, E, I, R
S, E, I, R = solution.T

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.title('SEIR Model Simulation')
plt.grid(True)
plt.show()
