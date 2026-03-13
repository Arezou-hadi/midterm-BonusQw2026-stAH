import numpy as np
import matplotlib.pyplot as plt

# --- 1. Constants and Given Data ---
G = 6.67430e-11  # m^3/kg/s^2
# Mass anomalies [cite: 146]
masses = np.array([8.54e5, 3.26e6, 5.21e6]) 
# Positions: {x, y, z} [cite: 146]
# Note: z is negative for subsurface
pos = np.array([
    [76, -86, -26],
    [-34, -26, -7],
    [-94, 36, -10]
])

M_actual = np.sum(masses)  # Actual total mass 

# --- 2. Define the Grid ---
# We start with a large domain to find the 1% boundary [cite: 149]
spacing = 5.0  # [cite: 149]
limit = 1000   # Initial search limit in meters
x = np.arange(-limit, limit + spacing, spacing)
y = np.arange(-limit, limit + spacing, spacing)
X, Y = np.meshgrid(x, y)

def compute_gz(X, Y, masses, pos):
    gz_total = np.zeros_like(X)
    for i in range(len(masses)):
        # Calculate distance r to point on surface (z=0) 
        dx = X - pos[i, 0]
        dy = Y - pos[i, 1]
        dz = 0 - pos[i, 2] # Vertical distance
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        # Vertical gravity effect 
        gz_total += (G * masses[i] * np.abs(dz)) / (r**3)
    return gz_total

gz = compute_gz(X, Y, masses, pos)
gz_max = np.max(gz)

# --- 3. Determine Radius for 1% Condition ---
# Find coordinates where gz > 0.01 * gz_max [cite: 149]
mask = gz >= (0.01 * gz_max)
if np.any(mask):
    indices = np.argwhere(mask)
    y_idx, x_idx = indices[:, 0], indices[:, 1]
    x_needed = x[x_idx]
    y_needed = y[y_idx]
    # Radius r needed to enclose this region [cite: 150]
    r_needed = np.max(np.sqrt(x_needed**2 + y_needed**2))
else:
    r_needed = limit

# --- 4. Excess Mass Formula  ---
# Summing gravity over the area: M_hat = (1 / 2piG) * sum(gz * dA)
dA = spacing**2
M_hat = (1 / (2 * np.pi * G)) * np.sum(gz[mask]) * dA 

# --- 5. Error Quantification  ---
error_magnitude = np.abs(M_actual - M_hat)
error_percent = (error_magnitude / M_actual) * 100

# --- 6. Visualization ---
plt.figure(figsize=(10, 8))
cp = plt.contourf(X, Y, gz * 1e6, levels=50, cmap='viridis') # in microGals
plt.colorbar(cp, label='gz (μGal)')
plt.title(f'Vertical Gravity Effect gz(x,y)\nEstimated Mass: {M_hat:.2e} kg (Error: {error_percent:.2f}%)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.xlim(-r_needed, r_needed)
plt.ylim(-r_needed, r_needed)
plt.grid(alpha=0.3)
plt.show()

print(f"Actual Mass: {M_actual:.2e} kg")
print(f"Estimated Mass: {M_hat:.2e} kg")
print(f"Required Survey Radius: {r_needed:.2f} m")
print(f"Estimation Error: {error_percent:.2f}%")