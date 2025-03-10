/**
 * Hybrid equation that uses the largest intersection point:
 * - y = x (when x <= 1552.91)
 * - y = -8076.3073 + 1348.0510 * log(x + 402.8573) + -14.8968 * sqrt(x) (when x > 1552.91)
 * 
 * @param {number} x - Input value
 * @returns {number} Output value according to the hybrid equation
 */
function hybridEquation(x) {
    // Largest intersection point (where x = y on the logarithmic curve)
    const xIntersect = 1552.91;
    
    // Parameters for the enhanced logarithmic model
    const a = -8076.3073;
    const b = 1348.0510;
    const c = 402.8573;
    const d = -14.8968;
    
    // Define the logarithmic function
    function logFunc(x) {
        return a + b * Math.log(x + c) + d * Math.sqrt(x);
    }
    
    // Apply the appropriate part of the hybrid equation
    if (x <= xIntersect) {
        return x;  // Linear: y = x
    } else {
        return logFunc(x);  // Logarithmic
    }
}
