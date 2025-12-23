"""
Example: Gradient Descent Energy Minimisation with Plotting
"""

import matplotlib.pyplot as plt
from random_walk.core.energy import harmonic
from random_walk.core.optimiser import GradientDescent
from random_walk.core.call_backs import TrajectoryRecorder, PrintEnergy

# -----------------------------
# Set up the energy model
# -----------------------------
energy_model = harmonic(k=1.0)
x0 = [2.0, -1.0]  # initial position

# -----------------------------
# Set up callbacks
# -----------------------------
recorder = TrajectoryRecorder()
callbacks = [recorder, PrintEnergy(every=1)]

# -----------------------------
# Run gradient descent
# -----------------------------
gd = GradientDescent(energy_model=energy_model, step_size=0.1, callbacks=callbacks)
trajectory = gd.minimize(x0, max_iter=100)

# -----------------------------
# Extract recorded data
# -----------------------------
energies = recorder.get_energies()
traj = recorder.get_trajectory()

print("Final position:", trajectory[-1])

# -----------------------------
# Plot energy vs step
# -----------------------------
plt.figure(figsize=(8, 4))
plt.plot(energies, marker='o')
plt.xlabel("Step")
plt.ylabel("Energy")
plt.title("Gradient Descent Energy Minimisation")
plt.grid(True)
plt.tight_layout()
plt.savefig("harmonic_minimisation.png")  # saves figure
plt.show()
