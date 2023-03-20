# xapmal

MLCompute package from Apple seems to be too complicated to use. This package hopefully helps you to jumpstart your ML projects in Swift.

``` swift

// xapmal allow you to generate both MLCTensor and MLCTensor data in one structure xm.Tensor.
let x1 = xm.Tensor([5.0])
let x2 = xm.Tensor([2.0])

// Addition operator is working for xapmal tensors. Result is MLCTensor for now.
let y = x1 + x2

```