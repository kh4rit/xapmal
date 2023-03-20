import Foundation
import MLCompute

let descriptor: MLCTensorDescriptor = MLCTensorDescriptor(shape: [1,1], dataType: .float32)!
var x1: MLCTensor = MLCTensor(descriptor: descriptor, fillWithData: 2.0)
print(x1.data!)
