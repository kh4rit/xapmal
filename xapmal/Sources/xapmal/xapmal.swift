import MLCompute

@available(macOS 11.0, *)
public struct xm {
    public private(set) var text = "Hello, World!"

    public init() {
    }
    
    public static var backlink = [MLCTensor: Tensor]()
    
    public static let graph = MLCGraph()
    public static let device = MLCDevice()
    
    public class Tensor {
        public let mlcTensor: MLCTensor
        public var mlcTensorData: MLCTensorData?
        public var value: [Float]?
        public let mlcLayer: MLCLayer?
        
        public func evaluate() {
            assert(self.mlcLayer != nil, "ERROR: Layer is not defined")
            
            // Create a training graph and configure it with the graph.
            let trainingGraph = xm.createTrainingGraph(lossLayer: nil, optimizer: nil)
            trainingGraph.compile(device: device)
            
            let (inputs, datas) = self.getAllInputsAndData()
            
            // add inputs to the training graph
            trainingGraph.addInputs(inputs, lossLabels: nil)
            trainingGraph.addOutputs(["output": self.mlcTensor])
            
            trainingGraph.execute(inputsData: datas, lossLabelsData: nil, lossLabelWeightsData: nil, batchSize: 0, completionHandler: { (tensor, error, time) in
                guard let mlcTensor = tensor else { print(error!); return }
                
                let xmTensor = xm.Tensor(mlcTensorWithData: mlcTensor)
                
                self.mlcTensorData = xmTensor.mlcTensorData
                
                self.value = xmTensor.value
            })
        }
        
        func getAllInputsAndData() -> ([String: MLCTensor], [String: MLCTensorData]) {
            let tensors = xm.graph.sourceTensors(for: self.mlcLayer!)
            var inputs = [String: MLCTensor]()
            var datas = [String: MLCTensorData]()
            let lenght = tensors.count - 1
            for i in 0...lenght {
                inputs["input_\(i)"] = tensors[i]
                datas["input_\(i)"] = xm.Tensor(mlcTensorWithData: tensors[i]).mlcTensorData
            }
            return (inputs, datas)
        }
        
        public init(_ value: [Float]) {
            self.value = value
            self.mlcTensorData = MLCTensorData(immutableBytesNoCopy: UnsafeRawPointer(self.value!), length: self.value!.count * MemoryLayout<Float>.size)
            self.mlcTensor = MLCTensor(shape: [value.count], data: self.mlcTensorData!, dataType: .float32)
            self.mlcLayer = nil
            xm.backlink[self.mlcTensor] = self
        }
        
        public init(mlcTensor: MLCTensor, mlcLayer: MLCLayer) {
            self.value = nil
            self.mlcTensorData = nil
            self.mlcTensor = mlcTensor
            self.mlcLayer = mlcLayer
            xm.backlink[self.mlcTensor] = self
        }
        
        public init(mlcTensorWithData mlcTensor: MLCTensor) {
            // Probably it is not a good idea, better to construct tensor in it's methods. TODO: remove.
            let value = xm.getFloatArray(mlcTensor)!
            self.value = value
            self.mlcTensorData = MLCTensorData(immutableBytesNoCopy: UnsafeRawPointer(self.value!), length: self.value!.count * MemoryLayout<Float>.size)
            self.mlcTensor = mlcTensor
            self.mlcLayer = nil
            xm.backlink[self.mlcTensor] = self
        }
    }
    
    // Looks interesting, but not working
//    public static func dataToMLC(_ data: Data) -> MLCTensorData {
//        let tensorDataBytes = data.withUnsafeBytes { $0.baseAddress }
//        let tensorDataLength = data.count
//        return MLCTensorData(immutableBytesNoCopy: tensorDataBytes!, length: tensorDataLength)
//    }
    
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
    static func +(lhs: xm.Tensor, rhs: xm.Tensor) -> xm.Tensor {
        let layer = MLCArithmeticLayer(operation: .add)
        let tensor = xm.Tensor(mlcTensor: xm.graph.node(with: layer, sources: [lhs.mlcTensor, rhs.mlcTensor])!, mlcLayer: layer)
        return tensor
    }
}
