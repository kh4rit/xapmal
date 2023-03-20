import MLCompute

@available(macOS 11.0, *)
public struct xm {
    public private(set) var text = "Hello, World!"

    public init() {
    }
    
    public static let graph = MLCGraph()
    
    public struct Tensor {
        public let MLCTensor: MLCTensor
        public let MLCTensorData: MLCTensorData
        public let value: [Float]
        
        public init(_ value: [Float]) {
            self.value = value
            self.MLCTensor = MLCompute.MLCTensor(shape: [value.count], dataType: .float32)
            self.MLCTensorData = MLCompute.MLCTensorData(immutableBytesNoCopy: UnsafeRawPointer(self.value), length: self.value.count * MemoryLayout<Float>.size)
        }
    }
    
    public static func createTrainingGraph(lossLayer: MLCLayer?, optimizer: MLCOptimizer?) -> MLCTrainingGraph {
        return MLCTrainingGraph(graphObjects: [self.graph], lossLayer: lossLayer, optimizer: optimizer)
    }
    
    public static func getFloatArray(_ mlcTensor: MLCTensor) -> [Float]? {
        guard let data = mlcTensor.data else { return nil }
        
        let floatArray = data.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) -> [Float] in
            let floatBuffer = pointer.bindMemory(to: Float.self)
            return Array(floatBuffer)
        }
        
        return floatArray
    }
}

// adding support of operators
@available(macOS 11.0, *)
public extension xm.Tensor {
    static func +(lhs: xm.Tensor, rhs: xm.Tensor) -> MLCTensor {
        return xm.graph.node(with: MLCArithmeticLayer(operation: .add), sources: [lhs.MLCTensor, rhs.MLCTensor])!
    }
}
