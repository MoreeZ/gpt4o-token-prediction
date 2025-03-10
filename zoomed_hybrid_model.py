import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Define the enhanced logarithmic function
def logarithmic_func(x, a, b, c, d):
    """Enhanced logarithmic model: y = a + b * log(x + c) + d * sqrt(x)"""
    return a + b * np.log(x + c) + d * np.sqrt(x)

# Parameters for the enhanced logarithmic model
a = -8076.3073
b = 1348.0510
c = 402.8573
d = -14.8968

# Find the largest intersection point where x = y
def find_largest_intersection(max_x=50000):
    # Define the function to find where x = y (logarithmic_func(x) - x = 0)
    def func_to_solve(x):
        return logarithmic_func(x, a, b, c, d) - x
    
    intersections = []
    
    # Search in different ranges with different step sizes
    ranges = [
        (0, 10, 20),           # Small values
        (10, 100, 20),         # Medium-small values
        (100, 1000, 20),       # Medium values
        (1000, 10000, 20),     # Medium-large values
        (10000, max_x, 20)     # Large values
    ]
    
    for start, end, steps in ranges:
        for guess in np.linspace(start, end, steps):
            try:
                result = optimize.root(func_to_solve, guess)
                if result.success:
                    x_intersect = result.x[0]
                    # Check if it's a valid solution (within our range and x ≈ y)
                    if start <= x_intersect <= end and abs(func_to_solve(x_intersect)) < 0.1:
                        y_intersect = logarithmic_func(x_intersect, a, b, c, d)
                        # Add to our list if it's not too close to an existing point
                        if not any(abs(x - x_intersect) < 1 for x, _ in intersections):
                            intersections.append((x_intersect, y_intersect))
            except:
                continue
    
    # Sort by x value and return the largest
    if intersections:
        return sorted(intersections)[-1]
    return None

# Find the largest intersection point
intersection = find_largest_intersection()
if intersection:
    x_intersect, y_intersect = intersection
    print(f"Largest intersection point: x = {x_intersect:.2f}, y = {y_intersect:.2f}")
else:
    # Use a default value if no intersection found
    x_intersect, y_intersect = 1552.91, 1552.91
    print(f"Using default intersection point: x = {x_intersect:.2f}, y = {y_intersect:.2f}")

# Find the maximum point of the logarithmic function
def find_max_log_point(max_x=50000):
    # Create a dense grid of x values
    x_values = np.linspace(0, max_x, 10000)
    y_values = logarithmic_func(x_values, a, b, c, d)
    
    # Find the maximum y value and corresponding x value
    max_idx = np.argmax(y_values)
    max_x = x_values[max_idx]
    max_y = y_values[max_idx]
    
    return max_x, max_y

max_x, max_y = find_max_log_point()
print(f"Maximum point on logarithmic curve: x = {max_x:.2f}, y = {max_y:.2f}")

# Define the hybrid model with the largest intersection point
def hybrid_model(x):
    if isinstance(x, (list, np.ndarray)):
        result = np.zeros_like(x, dtype=float)
        mask_linear = x <= x_intersect
        mask_log = x > x_intersect
        
        result[mask_linear] = x[mask_linear]  # Linear: y = x
        result[mask_log] = logarithmic_func(x[mask_log], a, b, c, d)
        return result
    else:
        if x <= x_intersect:
            return x  # Linear: y = x
        else:
            return logarithmic_func(x, a, b, c, d)

# Original data points
x_data = np.array([0, 1945, 4874, 10627, 22761, 29845, 38911])
y_data = np.array([0, 1836, 2468, 2392, 3930, 3272, 2945])

# Create a clean, elegant plot
plt.figure(figsize=(12, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# Set the x-axis limit to include the maximum point on the logarithmic curve
x_limit = max(max_x * 1.1, 40000)  # Ensure we include all data points

# Generate points for plotting
x_linear = np.linspace(0, x_intersect, 200)
y_linear = x_linear  # y = x

x_log = np.linspace(x_intersect, x_limit, 800)
y_log = logarithmic_func(x_log, a, b, c, d)

# Plot the two segments with a clean, modern look
plt.plot(x_linear, y_linear, color='#FF5733', linewidth=3, label=f'Linear (y = x) for x ≤ {x_intersect:.0f}')
plt.plot(x_log, y_log, color='#33A1FF', linewidth=3, label=f'Logarithmic for x > {x_intersect:.0f}')

# Plot the original data points
plt.scatter(x_data, y_data, color='#3D3D3D', s=100, alpha=0.8, edgecolor='white', linewidth=1.5, label='Original Data Points')

# Mark the transition point
plt.plot(x_intersect, y_intersect, 'o', color='#2ECC71', markersize=12, zorder=10)
plt.annotate(f'Transition Point\n({x_intersect:.0f}, {y_intersect:.0f})',
             xy=(x_intersect, y_intersect), xytext=(x_intersect-4000, y_intersect+500),
             arrowprops=dict(facecolor='#2ECC71', shrink=0.05, width=1.5, alpha=0.8),
             fontsize=12, color='#2ECC71', fontweight='bold')

# Mark the maximum point on the logarithmic curve
plt.plot(max_x, max_y, 'o', color='#9B59B6', markersize=12, zorder=10)
plt.annotate(f'Maximum Point\n({max_x:.0f}, {max_y:.0f})',
             xy=(max_x, max_y), xytext=(max_x-4000, max_y-500),
             arrowprops=dict(facecolor='#9B59B6', shrink=0.05, width=1.5, alpha=0.8),
             fontsize=12, color='#9B59B6', fontweight='bold')

# Add a clean text box for the equation
equation_text = f"""Hybrid Model Equation:
y = x                                                   if x ≤ {x_intersect:.0f}
y = {a:.2f} + {b:.2f} × log(x + {c:.2f}) {d:.2f} × √x   if x > {x_intersect:.0f}"""

plt.text(0.5, 0.03, equation_text, transform=plt.gcf().transFigure,
         fontsize=12, ha='center', va='center',
         bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.8', 
                   edgecolor='#CCCCCC'))

# Customize the plot
plt.xlabel('Input (x)', fontsize=14, fontweight='bold')
plt.ylabel('Output (y)', fontsize=14, fontweight='bold')
plt.title('Hybrid Model: Linear + Logarithmic', fontsize=18, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12, loc='upper right', framealpha=0.95)

# Set axis limits to focus on the relevant range
y_limit = max_y * 1.1  # Add 10% margin above the maximum point
plt.xlim(0, x_limit)
plt.ylim(0, y_limit)

# Add a small inset to show the transition point
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(plt.gca(), width="30%", height="30%", loc='lower right')
axins.plot(np.linspace(x_intersect-500, x_intersect, 50), np.linspace(x_intersect-500, x_intersect, 50), 
           color='#FF5733', linewidth=2)
axins.plot(np.linspace(x_intersect, x_intersect+500, 50), 
           logarithmic_func(np.linspace(x_intersect, x_intersect+500, 50), a, b, c, d), 
           color='#33A1FF', linewidth=2)
axins.plot(x_intersect, y_intersect, 'o', color='#2ECC71', markersize=8)

# Set the limits for the inset
axins.set_xlim(x_intersect-500, x_intersect+500)
axins.set_ylim(y_intersect-500, y_intersect+500)
axins.grid(True, alpha=0.3, linestyle='--')
axins.set_title('Zoom at Transition Point', fontsize=10, fontweight='bold')

plt.tight_layout(rect=[0, 0.07, 1, 0.95])
plt.savefig('zoomed_hybrid_model.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved as 'zoomed_hybrid_model.png'")

# Create a JavaScript function for the final hybrid model
js_code = f"""/**
 * Hybrid equation that uses the largest intersection point:
 * - y = x (when x <= {x_intersect:.2f})
 * - y = {a:.4f} + {b:.4f} * log(x + {c:.4f}) + {d:.4f} * sqrt(x) (when x > {x_intersect:.2f})
 * 
 * @param {{number}} x - Input value
 * @returns {{number}} Output value according to the hybrid equation
 */
function hybridEquation(x) {{
    // Largest intersection point (where x = y on the logarithmic curve)
    const xIntersect = {x_intersect:.2f};
    
    // Parameters for the enhanced logarithmic model
    const a = {a:.4f};
    const b = {b:.4f};
    const c = {c:.4f};
    const d = {d:.4f};
    
    // Define the logarithmic function
    function logFunc(x) {{
        return a + b * Math.log(x + c) + d * Math.sqrt(x);
    }}
    
    // Apply the appropriate part of the hybrid equation
    if (x <= xIntersect) {{
        return x;  // Linear: y = x
    }} else {{
        return logFunc(x);  // Logarithmic
    }}
}}
"""

# Save the JavaScript function to a file
with open('zoomed_hybrid_equation.js', 'w', encoding='utf-8') as f:
    f.write(js_code)

print("\nCreated JavaScript function in 'zoomed_hybrid_equation.js'")
print("\nFinal Hybrid Model Equation:")
print("y = {")
print(f"    x                                                      if x <= {x_intersect:.2f}")
print(f"    {a:.4f} + {b:.4f} * log(x + {c:.4f}) + {d:.4f} * sqrt(x)   if x > {x_intersect:.2f}")
print("}")
