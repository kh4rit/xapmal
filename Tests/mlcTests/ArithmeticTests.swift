import XCTest
import MLCompute

final class ArithmeticTests: XCTestCase {

    func testSimpleSum() throws {
        // XCTest Documenation
        // https://developer.apple.com/documentation/xctest

        // Defining Test Cases and Test Methods
        // https://developer.apple.com/documentation/xctest/defining_test_cases_and_test_methods

        // Create two single-element tensors with float values.
        let x1: [Float] = [4.0]
        let x1Data = MLCTensorData(immutableBytesNoCopy: UnsafeRawPointer(x1), length: x1.count * MemoryLayout<Float>.size)
        let x1Tensor = MLCTensor(shape: [1], data: x1Data, dataType: .float32)
        let x2: [Float] = [2.0]
        let x2Data = MLCTensorData(immutableBytesNoCopy: UnsafeRawPointer(x2), length: x2.count * MemoryLayout<Float>.size)
        let x2Tensor = MLCTensor(shape: [1], data: x2Data, dataType: .float32)
        
        // Create a computation graph that adds the two tensors.
        let graph = MLCGraph()
        let sumNode = graph.node(with: MLCArithmeticLayer(operation: .add), sources: [x1Tensor, x2Tensor])!
        
        // Create a training graph and configure it with the graph.
        let trainingGraph = MLCTrainingGraph(graphObjects: [graph], lossLayer: nil, optimizer: nil)
        
        // Compile the training graph for a specific device.
        let device = MLCDevice()
            
        trainingGraph.compile(device: device)
        
        // Execute the training graph.
        trainingGraph.addInputs(["input_1": x1Tensor, "input_2": x2Tensor], lossLabels: nil)
        trainingGraph.addOutputs(["output": sumNode])

        trainingGraph.execute(inputsData: ["input_1": x1Data, "input_2": x2Data], lossLabelsData: nil, lossLabelWeightsData: nil, batchSize: 0, completionHandler: { (tensor, error, time) in
            guard let data = tensor?.data else { print(error!); return }
            
            let floatArray = data.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) -> [Float] in
                let floatBuffer = pointer.bindMemory(to: Float.self)
                return Array(floatBuffer)
            }

            // print("Result:", floatArray)
            XCTAssertEqual(floatArray, [6.0])
        })
    }

    func testSumArray() throws {
        // XCTest Documenation
        // https://developer.apple.com/documentation/xctest

        // Defining Test Cases and Test Methods
        // https://developer.apple.com/documentation/xctest/defining_test_cases_and_test_methods

        // Create two single-element tensors with float values.
        let x1: [Float] = [123.0, 234.0, 345.0, 456.0]
        let x1Data = MLCTensorData(immutableBytesNoCopy: UnsafeRawPointer(x1), length: x1.count * MemoryLayout<Float>.size)
        let x1Tensor = MLCTensor(shape: [4], data: x1Data, dataType: .float32)
        let x2: [Float] = [321.0, 432.0, 543.0, 654.0]
        let x2Data = MLCTensorData(immutableBytesNoCopy: UnsafeRawPointer(x2), length: x2.count * MemoryLayout<Float>.size)
        let x2Tensor = MLCTensor(shape: [4], data: x2Data, dataType: .float32)
        
        // Create a computation graph that adds the two tensors.
        let graph = MLCGraph()
        let sumNode = graph.node(with: MLCArithmeticLayer(operation: .add), sources: [x1Tensor, x2Tensor])!
        
        // Create a training graph and configure it with the graph.
        let trainingGraph = MLCTrainingGraph(graphObjects: [graph], lossLayer: nil, optimizer: nil)
        
        // Compile the training graph for a specific device.
        let device = MLCDevice()
            
        trainingGraph.compile(device: device)
        
        // Execute the training graph.
        trainingGraph.addInputs(["input_1": x1Tensor, "input_2": x2Tensor], lossLabels: nil)
        trainingGraph.addOutputs(["output": sumNode])

        trainingGraph.execute(inputsData: ["input_1": x1Data, "input_2": x2Data], lossLabelsData: nil, lossLabelWeightsData: nil, batchSize: 0, completionHandler: { (tensor, error, time) in
            guard let data = tensor?.data else { print(error!); return }
            
            let floatArray = data.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) -> [Float] in
                let floatBuffer = pointer.bindMemory(to: Float.self)
                return Array(floatBuffer)
            }

            // print("Result:", floatArray)
            XCTAssertEqual(floatArray, [444.0, 666.0, 888.0, 1110.0])
        })
    }

}
