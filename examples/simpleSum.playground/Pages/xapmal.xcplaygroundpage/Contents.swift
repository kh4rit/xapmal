import Foundation
import MLCompute
import xapmal

// Create two single-element tensors with float values.
let x1 = xm.Tensor([5.0])
let x2 = xm.Tensor([2.0])

let y = x1 + x2

// Create a training graph and configure it with the graph.
let trainingGraph = xm.createTrainingGraph(lossLayer: nil, optimizer: nil)

// Compile the training graph for a specific device.
let device = MLCDevice()
    
trainingGraph.compile(device: device)

// Execute the training graph.
trainingGraph.addInputs(["input_1": x1.MLCTensor, "input_2": x2.MLCTensor], lossLabels: nil)
trainingGraph.addOutputs(["output": y])

trainingGraph.execute(inputsData: ["input_1": x1.MLCTensorData, "input_2": x2.MLCTensorData], lossLabelsData: nil, lossLabelWeightsData: nil, batchSize: 0, completionHandler: { (tensor, error, time) in
    guard let tensor = tensor else { print(error!); return }
    
    guard let floatArray = xm.getFloatArray(tensor) else { print("ERROR: tensor data couldn't be converted to Float array"); return }
    
    print("Result:", floatArray[0])
})
