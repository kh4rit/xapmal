import MLCompute

@available(macOS 12.0, *)
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
            let trainingGraph = MLCTrainingGraph(graphObjects: [xm.graph], lossLayer: nil, optimizer: nil)
            trainingGraph.compile(device: device)
            
            let (inputs, datas) = self.getAllInputsAndData()
            
            // add inputs to the training graph
            trainingGraph.addInputs(inputs, lossLabels: nil)
            trainingGraph.addOutputs(["output": self.mlcTensor])
            
            trainingGraph.execute(inputsData: datas, lossLabelsData: nil, lossLabelWeightsData: nil, batchSize: 0, completionHandler: { (tensor, error, time) in
                guard let mlcTensor = tensor else { print(error!); return }
                
                let value = xm.getFloatArray(mlcTensor)!
                self.value = value
                self.mlcTensorData = MLCTensorData(immutableBytesNoCopy: UnsafeRawPointer(self.value!), length: self.value!.count * MemoryLayout<Float>.size)
            })
        }
        
        func getAllInputsAndData() -> ([String: MLCTensor], [String: MLCTensorData]) {
            let tensors = xm.graph.sourceTensors(for: self.mlcLayer!)
            var inputs = [String: MLCTensor]()
            var datas = [String: MLCTensorData]()
            let lenght = tensors.count - 1
            for i in 0...lenght {
                inputs["input_\(i)"] = tensors[i]
                datas["input_\(i)"] = xm.backlink[tensors[i]]!.mlcTensorData!
            }
            return (inputs, datas)
        }
        
        public init(_ value: [Float]) {
            self.value = value
            let elemType = xm.elementType(of: value)
            let (mlcType, memorySize) = xm.mlcDataType(of: elemType)
            let numOfElem = xm.countElements(of: value)
            self.mlcTensorData = MLCTensorData(immutableBytesNoCopy: UnsafeRawPointer(self.value!), length: numOfElem * memorySize)
            self.mlcTensor = MLCTensor(shape: shape(of: value), data: self.mlcTensorData!, dataType: mlcType)
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
    }
    
    // Looks interesting, but not working
//    public static func dataToMLC(_ data: Data) -> MLCTensorData {
//        let tensorDataBytes = data.withUnsafeBytes { $0.baseAddress }
//        let tensorDataLength = data.count
//        return MLCTensorData(immutableBytesNoCopy: tensorDataBytes!, length: tensorDataLength)
//    }
    
    public static func getFloatArray(_ mlcTensor: MLCTensor) -> [Float]? {
        guard let data = mlcTensor.data else { return nil }
        
        let floatArray = data.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) -> [Float] in
            let floatBuffer = pointer.bindMemory(to: Float.self)
            return Array(floatBuffer)
        }
        
        return floatArray
    }
    
    // beautiful fuction from GPT-4 to calculate a shape of array.
    static func shape(of array: Any) -> [Int] {
        if let nestedArray = array as? [Any] {
            return [nestedArray.count] + shape(of: nestedArray.first ?? [])
        } else {
            return []
        }
    }
    
    static func countElements(of array: Any) -> Int {
        if let nestedArray = array as? [Any] {
            if nestedArray.isEmpty {
                return 0
            }
            let firstElement = nestedArray.first
            if firstElement is [Any] {
                return nestedArray.reduce(0) { (sum, element) in
                    sum + countElements(of: element)
                }
            } else {
                return nestedArray.count
            }
        }
        return 0
    }
    
    static func elementType(of array: Any) -> Any.Type? {
        if let nestedArray = array as? [Any], !nestedArray.isEmpty {
            return elementType(of: nestedArray.first ?? [])
        } else {
            return type(of: array)
        }
    }
    
    static func mlcDataType(of elementType: Any.Type?) -> (MLCDataType, Int) {
        guard let type = elementType else { fatalError("Could not define the MLC Data Type for element type \(String(describing: elementType))") }
        switch type {
        case is Int8.Type:
            return (.int8, size: MemoryLayout<Float>.size)
        case is Int32.Type:
            return (.int32, size: MemoryLayout<Int32>.size)
        // Int is 64 bit only in 64 bit systems. All modern are 64, so we assume it is.
        case is Int.Type, is Int64.Type:
            return (.int64, size: MemoryLayout<Int64>.size)
        case is UInt8.Type:
            return (.uint8, size: MemoryLayout<UInt8>.size)
        case is Float.Type:
            return (.float32, size: MemoryLayout<Float>.size)
        case is Bool.Type:
            return (.boolean, size: MemoryLayout<Bool>.size)
        default:
            fatalError("Could not define the MLC Data Type for element type \(type)")
        }
    }
    
}

// adding support of operators
@available(macOS 12.0, *)
public extension xm.Tensor {
    static func +(lhs: xm.Tensor, rhs: xm.Tensor) -> xm.Tensor {
        let layer = MLCArithmeticLayer(operation: .add)
        let tensor = xm.Tensor(mlcTensor: xm.graph.node(with: layer, sources: [lhs.mlcTensor, rhs.mlcTensor])!, mlcLayer: layer)
        return tensor
    }
}
