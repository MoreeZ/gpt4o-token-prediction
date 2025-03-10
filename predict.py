import numpy as np

def predict_logarithmic(x):
    """
    Standard logarithmic model: y = a + b * log(x + c)
    
    Parameters:
    x (float or array): Input value(s) to predict output for
    
    Returns:
    float or array: Predicted output value(s)
    """
    a = -1975.2872
    b = 509.8275
    c = 48.0617
    
    return a + b * np.log(x + c)

def predict_enhanced(x):
    """
    Enhanced logarithmic model: y = a + b * log(x + c) + d * sqrt(x)
    
    Parameters:
    x (float or array): Input value(s) to predict output for
    
    Returns:
    float or array: Predicted output value(s)
    """
    a = -8076.3073
    b = 1348.0510
    c = 402.8573
    d = -14.8968
    
    return a + b * np.log(x + c) + d * np.sqrt(x)

if __name__ == "__main__":
    # Example usage
    test_inputs = [0, 1945, 4874, 10627, 22761, 29845, 38911, 50000]
    
    print("Enhanced Logarithmic Model Predictions")
    print("======================================")
    print(f"{'Input':<10} {'Predicted Output':<20}")
    print("-" * 30)
    
    for x in test_inputs:
        y_pred = predict_enhanced(x)
        print(f"{x:<10} {y_pred:.2f}")
    
    # Interactive mode
    print("\nInteractive Mode")
    print("===============")
    print("Enter input values (or 'q' to quit):")
    
    while True:
        user_input = input("Input value: ")
        if user_input.lower() == 'q':
            break
        
        try:
            x = float(user_input)
            y_pred = predict_enhanced(x)
            print(f"Predicted output: {y_pred:.2f}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")
