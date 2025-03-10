import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Data points (updated with all provided data)
x_data = np.array([0, 1945, 4874, 10627, 22761, 29845, 38911])
y_data = np.array([0, 1836, 2468, 2392, 3930, 3272, 2945])

print("Fitting logarithmic models to the data...")
print("-" * 50)

# Define the logarithmic function: y = a + b * log(x + c)
def logarithmic_func(x, a, b, c):
    # Adding a small constant to x to avoid log(0)
    return a + b * np.log(x + c)

# Alternative logarithmic function with square root term
def alt_log_func(x, a, b, c, d):
    # More complex logarithmic function with additional parameters
    return a + b * np.log(x + c) + d * np.sqrt(x)

# Initial parameter guesses
initial_guess = [0, 500, 1]  # a, b, c

# Curve fitting for standard logarithmic function
try:
    # Use curve_fit to find the optimal parameters
    params, params_covariance = optimize.curve_fit(
        logarithmic_func, 
        x_data, 
        y_data, 
        p0=initial_guess,
        maxfev=10000  # Increase the maximum number of function evaluations
    )
    
    a_fit, b_fit, c_fit = params
    print(f"MODEL 1: Standard Logarithmic Function")
    print(f"Fitted parameters: a = {a_fit:.4f}, b = {b_fit:.4f}, c = {c_fit:.4f}")
    print(f"Logarithmic function: y = {a_fit:.4f} + {b_fit:.4f} * log(x + {c_fit:.4f})")
    
    # Generate points for the fitted curve
    x_fit = np.linspace(0, max(x_data) * 1.1, 1000)
    y_fit = logarithmic_func(x_fit, a_fit, b_fit, c_fit)
    
    # Calculate R-squared to evaluate the goodness of fit
    y_pred = logarithmic_func(x_data, a_fit, b_fit, c_fit)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    ss_res = np.sum((y_data - y_pred)**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R-squared: {r_squared:.6f}")
    
    # Calculate Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((y_data - y_pred)**2))
    print(f"RMSE: {rmse:.4f}")
    
    # Plot the data points and the fitted curve
    plt.figure(figsize=(12, 7))
    plt.scatter(x_data, y_data, color='blue', label='Data points', s=80, alpha=0.7)
    plt.plot(x_fit, y_fit, color='red', linewidth=2, 
             label=f'Logarithmic fit: y = {a_fit:.2f} + {b_fit:.2f} * log(x + {c_fit:.2f})')
    
    # Add data point labels
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        plt.annotate(f'({x}, {y})', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    # Customize the plot
    plt.xlabel('Input (x)', fontsize=12)
    plt.ylabel('Output (y)', fontsize=12)
    plt.title('Logarithmic Line of Best Fit', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add equation and R-squared on the plot
    equation_text = f"y = {a_fit:.2f} + {b_fit:.2f} × ln(x + {c_fit:.2f})"
    plt.figtext(0.5, 0.01, f"{equation_text}\nR² = {r_squared:.4f}, RMSE = {rmse:.2f}", 
                ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('logarithmic_fit.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print the predicted values for the original inputs
    print("\nPredictions for original inputs:")
    print(f"{'Input':<10} {'Actual':<10} {'Predicted':<10} {'Error':<10}")
    print("-" * 40)
    for x, y_actual in zip(x_data, y_data):
        y_predicted = logarithmic_func(x, a_fit, b_fit, c_fit)
        error = y_predicted - y_actual
        print(f"{x:<10} {y_actual:<10} {y_predicted:.2f}{'':<2} {error:.2f}")
    
    # Function to predict output for a new input
    def predict_output(input_value):
        return logarithmic_func(input_value, a_fit, b_fit, c_fit)
    
    # Example of using the function for prediction
    print("\nPredict for new inputs:")
    test_inputs = [2000, 5000, 10000, 20000, 30000, 50000]
    for test_input in test_inputs:
        print(f"Input: {test_input}, Predicted Output: {predict_output(test_input):.2f}")
    
    # Create a simple function that users can call
    print("\nYou can use the following function in your code to make predictions:")
    print("def predict(x):")
    print(f"    return {a_fit:.4f} + {b_fit:.4f} * np.log(x + {c_fit:.4f})")
    
except RuntimeError as e:
    print(f"Error in curve fitting: {e}")

# Now try the alternative logarithmic function with square root term
print("\n" + "-" * 50)
print("MODEL 2: Enhanced Logarithmic Function with Square Root Term")
print("-" * 50)

try:
    alt_initial_guess = [0, 500, 1, 0.01]  # a, b, c, d
    params, _ = optimize.curve_fit(alt_log_func, x_data, y_data, p0=alt_initial_guess, maxfev=10000)
    a_fit, b_fit, c_fit, d_fit = params
    print(f"Fitted parameters: a = {a_fit:.4f}, b = {b_fit:.4f}, c = {c_fit:.4f}, d = {d_fit:.4f}")
    print(f"Enhanced function: y = {a_fit:.4f} + {b_fit:.4f} * log(x + {c_fit:.4f}) + {d_fit:.4f} * sqrt(x)")
    
    # Generate points for the fitted curve
    x_fit = np.linspace(0, max(x_data) * 1.1, 1000)
    y_fit = alt_log_func(x_fit, a_fit, b_fit, c_fit, d_fit)
    
    # Calculate R-squared
    y_pred = alt_log_func(x_data, a_fit, b_fit, c_fit, d_fit)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    ss_res = np.sum((y_data - y_pred)**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R-squared: {r_squared:.6f}")
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_data - y_pred)**2))
    print(f"RMSE: {rmse:.4f}")
    
    # Plot the data points and the fitted curve
    plt.figure(figsize=(12, 7))
    plt.scatter(x_data, y_data, color='blue', label='Data points', s=80, alpha=0.7)
    plt.plot(x_fit, y_fit, color='green', linewidth=2, label='Enhanced logarithmic fit')
    
    # Add data point labels
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        plt.annotate(f'({x}, {y})', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.xlabel('Input (x)', fontsize=12)
    plt.ylabel('Output (y)', fontsize=12)
    plt.title('Enhanced Logarithmic Line of Best Fit', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add equation and R-squared on the plot
    equation_text = f"y = {a_fit:.2f} + {b_fit:.2f} × ln(x + {c_fit:.2f}) + {d_fit:.4f} × √x"
    plt.figtext(0.5, 0.01, f"{equation_text}\nR² = {r_squared:.4f}, RMSE = {rmse:.2f}", 
                ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('logarithmic_fit_enhanced.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print the predicted values for the original inputs
    print("\nPredictions for original inputs (enhanced model):")
    print(f"{'Input':<10} {'Actual':<10} {'Predicted':<10} {'Error':<10}")
    print("-" * 40)
    for x, y_actual in zip(x_data, y_data):
        y_predicted = alt_log_func(x, a_fit, b_fit, c_fit, d_fit)
        error = y_predicted - y_actual
        print(f"{x:<10} {y_actual:<10} {y_predicted:.2f}{'':<2} {error:.2f}")
    
    # Function to predict output for a new input using the enhanced model
    def predict_output_enhanced(input_value):
        return alt_log_func(input_value, a_fit, b_fit, c_fit, d_fit)
    
    # Example of using the function for prediction
    print("\nPredict for new inputs (enhanced model):")
    test_inputs = [2000, 5000, 10000, 20000, 30000, 50000]
    for test_input in test_inputs:
        print(f"Input: {test_input}, Predicted Output: {predict_output_enhanced(test_input):.2f}")
    
    # Create a simple function that users can call
    print("\nYou can use the following function in your code to make predictions with the enhanced model:")
    print("def predict_enhanced(x):")
    print(f"    return {a_fit:.4f} + {b_fit:.4f} * np.log(x + {c_fit:.4f}) + {d_fit:.4f} * np.sqrt(x)")
    
    # Compare models
    print("\n" + "-" * 50)
    print("MODEL COMPARISON")
    print("-" * 50)
    if r_squared > 0.895666:  # R-squared from the standard model
        print(f"The enhanced logarithmic model with square root term provides a better fit (R² = {r_squared:.6f})")
    else:
        print(f"The standard logarithmic model provides a better fit (R² = 0.895666)")
    
except Exception as e:
    print(f"Alternative approach failed: {e}")
