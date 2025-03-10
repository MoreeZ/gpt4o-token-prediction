# GPT-4o Token Response Prediction Model

This project analyzes and predicts the relationship between input tokens (characters sent to ChatGPT) and output tokens (characters returned in responses) for the GPT-4o-2024-08-06 model. The analysis reveals a hybrid pattern where responses follow a linear relationship for smaller inputs and a logarithmic relationship for larger inputs.

## Data Context

The dataset represents actual input and output character counts from interactions with the GPT-4o-2024-08-06 model:

| Input Characters (x) | Output Characters (y) |
|----------------------|----------------------|
| 0                    | 0                    |
| 1945                 | 1836                 |
| 4874                 | 2468                 |
| 10627                | 2392                 |
| 22761                | 3930                 |
| 29845                | 3272                 |
| 38911                | 2945                 |

This data reveals an interesting pattern: for smaller inputs, the response size tends to match the input size (approximately 1:1 ratio), but for larger inputs, the response size grows logarithmically rather than linearly.

## Models Implemented

Three different models were developed to capture this relationship:

1. **Standard Logarithmic Model**: `y = a + b * log(x + c)`
2. **Enhanced Logarithmic Model**: `y = a + b * log(x + c) + d * sqrt(x)`
3. **Hybrid Model**: Combines linear function (y = x) with the enhanced logarithmic model at the largest intersection point

## Key Findings

### Standard Logarithmic Model

- **Formula**: y = -1975.29 + 509.83 * log(x + 48.06)
- **R²**: 0.895666
- **RMSE**: 375.9122

### Enhanced Logarithmic Model (with Square Root Term)

- **Formula**: y = -8076.31 + 1348.05 * log(x + 402.86) - 14.90 * sqrt(x)
- **R²**: 0.905363
- **RMSE**: 358.0174

### Hybrid Model

- **Formula**: 
  ```
  y = {
      x                                                      if x <= 1552.91
      -8076.3073 + 1348.0510 * log(x + 402.8573) - 14.8968 * sqrt(x)   if x > 1552.91
  }
  ```
- This model uses the largest intersection point (1552.91) where x = y on the logarithmic curve as the transition point.
- The hybrid model captures the observed behavior where GPT-4o responses match input length for shorter prompts but follow a logarithmic pattern for longer prompts.

## Practical Implications

This model provides insights into how GPT-4o allocates response tokens based on input size:

1. For inputs up to ~1553 characters, responses tend to match input length (y ≈ x)
2. For larger inputs, response length grows logarithmically, reaching a maximum around ~3260 characters
3. The maximum response length occurs at an input of approximately 31,943 characters

These insights can help optimize prompt engineering and manage expectations about response lengths when working with the GPT-4o model.

## Usage

### Running the Models

```bash
# For the logarithmic models
python logarithmic_fit.py

# For the hybrid model
python zoomed_hybrid_model.py
```

### Making Predictions

You can use the following functions in your code to predict GPT-4o response lengths:

#### Standard Logarithmic Model
```python
def predict(x):
    return -1975.2872 + 509.8275 * np.log(x + 48.0617)
```

#### Enhanced Logarithmic Model
```python
def predict_enhanced(x):
    return -8076.3073 + 1348.0510 * np.log(x + 402.8573) - 14.8968 * np.sqrt(x)
```

#### Hybrid Model
```python
def hybrid_model(x):
    x_intersect = 1552.91
    if x <= x_intersect:
        return x  # Linear: y = x
    else:
        return -8076.3073 + 1348.0510 * np.log(x + 402.8573) - 14.8968 * np.sqrt(x)
```

#### JavaScript Implementation
The project also includes JavaScript implementations of the hybrid model in `zoomed_hybrid_equation.js` for web applications.

## Project Structure

- `logarithmic_fit.py` - Implements and compares standard and enhanced logarithmic models
- `zoomed_hybrid_model.py` - Implements the hybrid model with automatic detection of the largest intersection point
- `predict.py` - Utility for making predictions with the models
- `zoomed_hybrid_equation.js` - JavaScript implementation of the hybrid model

## Dependencies

- numpy
- matplotlib
- scipy

Install dependencies with:
```bash
pip install -r requirements.txt
