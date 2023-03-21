import Foundation
import MLCompute
import xapmal

// Create two single-element tensors with float values.
let x1 = xm.Tensor([4.0])
let x2 = xm.Tensor([2.0])

// Addition operator is working for xapmal tensors.
let y = x1 + x2

// All MLCompute training staff is hidden behind a simple evaluate function
y.evaluate()

print("Result:", y.value!)
