import Foundation
import MLCompute
import xapmal

// Create two single-element tensors with float values.
let x1Value: [Float] = [4.0, 1.0]   // MLCompute doesn't support Float64 (aka Double),
let x2Value: [Float] = [2.0, 2.0]   // so we have to explicitly provide the Float (aka Float32) type
let x1 = xm.Tensor(x1Value)
let x2 = xm.Tensor(x2Value)

// Addition operator is working for xapmal tensors.
let y = x1 + x2

// All MLCompute training staff is hidden behind a simple evaluate function
y.evaluate()

print("Result:", y.value!)
