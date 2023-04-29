import XCTest
@testable import xapmal

final class ArithmeticTests: XCTestCase {
    func testSimpleSum() throws {
        // XCTest Documenation
        // https://developer.apple.com/documentation/xctest

        // Defining Test Cases and Test Methods
        // https://developer.apple.com/documentation/xctest/defining_test_cases_and_test_methods

        // Create two single-element tensors with float values.
		let x1Value: [Float] = [4.0]   // MLCompute doesn't support Float64 (aka Double),
		let x2Value: [Float] = [2.0]   // so we have to explicitly provide the Float (aka Float32) type
		let x1 = xm.Tensor(x1Value)
		let x2 = xm.Tensor(x2Value)
		
		// Addition operator is working for xapmal tensors.
		let y = x1 + x2
		
		// All MLCompute training staff is hidden behind a simple evaluate function
		y.evaluate()
		
		// print("Result:", y.value!)
		XCTAssertEqual(y.value!, [6.0])
    }

    func testSumArray() throws {
        // XCTest Documenation
        // https://developer.apple.com/documentation/xctest

        // Defining Test Cases and Test Methods
        // https://developer.apple.com/documentation/xctest/defining_test_cases_and_test_methods

        // Create two single-element tensors with float values.
		let x1Value: [Float] = [123.0, 234.0, 345.0, 456.0]   // MLCompute doesn't support Float64 (aka Double),
		let x2Value: [Float] = [321.0, 432.0, 543.0, 654.0]   // so we have to explicitly provide the Float (aka Float32) type
		let x1 = xm.Tensor(x1Value)
		let x2 = xm.Tensor(x2Value)
		
		// Addition operator is working for xapmal tensors.
		let y = x1 + x2
		
		// All MLCompute training staff is hidden behind a simple evaluate function
		y.evaluate()
		
		// print("Result:", y.value!)
		XCTAssertEqual(y.value!, [444.0, 666.0, 888.0, 1110.0])
    }
}
