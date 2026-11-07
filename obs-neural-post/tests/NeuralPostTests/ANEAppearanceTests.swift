import Foundation
import XCTest
import Metal
import CoreVideo
@testable import NeuralPost

final class ANEAppearanceTests:XCTestCase {
    func testTrainedAppearanceRequestAndAdmission() throws {
        guard let path=ProcessInfo.processInfo.environment["NEURAL_POST_ANE_MODEL"] else { throw XCTSkip("Set NEURAL_POST_ANE_MODEL for native Core ML test") }
        let device=try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let model=try ANEAppearance(url:URL(fileURLWithPath:path),device:device)
        var pixel:CVPixelBuffer?
        let attrs:[CFString:Any]=[kCVPixelBufferIOSurfacePropertiesKey:[:],kCVPixelBufferMetalCompatibilityKey:true]
        XCTAssertEqual(CVPixelBufferCreate(nil,1280,720,kCVPixelFormatType_32BGRA,attrs as CFDictionary,&pixel),kCVReturnSuccess)
        let input=try XCTUnwrap(pixel)
        CVPixelBufferLockBaseAddress(input,[])
        memset(CVPixelBufferGetBaseAddress(input),127,CVPixelBufferGetDataSize(input))
        let bytes=CVPixelBufferGetBaseAddress(input)!.assumingMemoryBound(to:UInt8.self)
        for y in 0..<720 { for x in 0..<1280 { bytes[y*CVPixelBufferGetBytesPerRow(input)+x*4+3]=255 } }
        CVPixelBufferUnlockBaseAddress(input,[])
        let first=model.start(input)
        // A second admitted job must never accumulate behind the first request.
        let second=model.start(input)
        let result=try first.take()
        XCTAssertEqual(result.buffer.length,16*108*192*2)
        XCTAssertGreaterThan(result.milliseconds,0)
        do { _=try second.take() } catch { XCTAssertEqual((error as NSError).code,26) }
        XCTAssertTrue([UInt32(2),UInt32(3)].contains(result.mode))
    }
    func testHybridPublicationResetAndProtectedHUD() throws {
        guard let path=ProcessInfo.processInfo.environment["NEURAL_POST_ANE_MODEL"],let weights=ProcessInfo.processInfo.environment["NEURAL_POST_WEIGHTS"] else { throw XCTSkip("Set hybrid model paths") }
        var pixel:CVPixelBuffer?
        let attrs:[CFString:Any]=[kCVPixelBufferIOSurfacePropertiesKey:[:],kCVPixelBufferMetalCompatibilityKey:true]
        XCTAssertEqual(CVPixelBufferCreate(nil,1280,720,kCVPixelFormatType_32BGRA,attrs as CFDictionary,&pixel),kCVReturnSuccess)
        let input=try XCTUnwrap(pixel)
        CVPixelBufferLockBaseAddress(input,[])
        memset(CVPixelBufferGetBaseAddress(input),127,CVPixelBufferGetDataSize(input))
        let bytes=CVPixelBufferGetBaseAddress(input)!.assumingMemoryBound(to:UInt8.self)
        for y in 0..<720 { for x in 0..<1280 { bytes[y*CVPixelBufferGetBytesPerRow(input)+x*4+3]=255 } }
        CVPixelBufferUnlockBaseAddress(input,[])
        let baseline=try PostPipeline(modelURL:URL(fileURLWithPath:weights))
        let hybrid=try PostPipeline(modelURL:URL(fileURLWithPath:weights),aneURL:URL(fileURLWithPath:path))
        defer { baseline.stop();hybrid.stop() }
        let plainDone=expectation(description:"plain SR"),hybridDone=expectation(description:"same-generation hybrid")
        var plain:ProcessedFrame?,output:ProcessedFrame?
        baseline.onComplete = { plain=$0;plainDone.fulfill() }
        hybrid.onComplete = { frame in
            XCTAssertEqual(frame.sequence,2,"Retired generation was published")
            output=frame;hybridDone.fulfill()
        }
        hybrid.submit(input,sequence:1,sourceSeconds:1);hybrid.reset()
        hybrid.submit(input,sequence:2,sourceSeconds:2)
        baseline.submit(input,sequence:2,sourceSeconds:2)
        wait(for:[plainDone,hybridDone],timeout:30)
        let a=try XCTUnwrap(plain),b=try XCTUnwrap(output)
        XCTAssertTrue(b.neural);XCTAssertTrue(b.appearance);XCTAssertGreaterThan(b.aneMS,0)
        XCTAssertEqual(b.sourceSeconds,2);XCTAssertEqual(hybrid.stats.failed,0)
        CVPixelBufferLockBaseAddress(a.pixels,.readOnly);CVPixelBufferLockBaseAddress(b.pixels,.readOnly)
        defer { CVPixelBufferUnlockBaseAddress(a.pixels,.readOnly);CVPixelBufferUnlockBaseAddress(b.pixels,.readOnly) }
        let ap=CVPixelBufferGetBaseAddress(a.pixels)!.assumingMemoryBound(to:UInt8.self),bp=CVPixelBufferGetBaseAddress(b.pixels)!.assumingMemoryBound(to:UInt8.self)
        let strideA=CVPixelBufferGetBytesPerRow(a.pixels),strideB=CVPixelBufferGetBytesPerRow(b.pixels)
        let decoder=try ANEAppearance(url:URL(fileURLWithPath:path),device:hybrid.device)
        let fields=try decoder.start(input).take()
        let fp=fields.buffer.contents().assumingMemoryBound(to:Float16.self)
        func delta(_ x:Int,_ y:Int)->[Double] {
            if Double(x)+0.5<1280*0.26 && Double(y)+0.5>=720*0.70 { return [0,0,0] }
            let qx=max(0,min(191,(Double(x)+0.5)*192/1280-0.5)),qy=max(0,min(107,(Double(y)+0.5)*108/720-0.5))
            let ax=Int(qx),ay=Int(qy),bx=min(ax+1,191),by=min(ay+1,107),fx=qx-Double(ax),fy=qy-Double(ay)
            var f=[Double](repeating:0,count:16)
            for k in 0..<16 {
                let base=k*192*108
                let top=Double(fp[base+ay*192+ax])*(1-fx)+Double(fp[base+ay*192+bx])*fx
                let bottom=Double(fp[base+by*192+ax])*(1-fx)+Double(fp[base+by*192+bx])*fx
                f[k]=top*(1-fy)+bottom*fy
            }
            let confidence=1/(1+exp(-f[15])),color=127.0/255
            var edit=[Double](repeating:0,count:3)
            for c in 0..<3 {
                let affine=0.125*((tanh(f[c*4])+tanh(f[c*4+1])+tanh(f[c*4+2]))*color+tanh(f[c*4+3]))
                edit[c]=confidence*(affine+(fields.mode==2 ? 0.08*tanh(f[12+c]) : 0))
            }
            var amount=1.0
            for e in edit { amount=min(amount,(e>=0 ? 1-color : color)/(abs(e)+1e-6)) }
            return edit.map { $0*max(0,min(1,amount)) }
        }
        for y in [0,1000,2159] { for x in [0,2000,3839] {
            if x<998 && y>=1512 { continue }
            let qx=max(0,min(1279,(Double(x)+0.5)/3-0.5)),qy=max(0,min(719,(Double(y)+0.5)/3-0.5))
            let ax=Int(qx),ay=Int(qy),bx=min(ax+1,1279),by=min(ay+1,719),fx=qx-Double(ax),fy=qy-Double(ay)
            let d0=delta(ax,ay),d1=delta(bx,ay),d2=delta(ax,by),d3=delta(bx,by)
            for c in 0..<3 {
                let d=(d0[c]*(1-fx)+d1[c]*fx)*(1-fy)+(d2[c]*(1-fx)+d3[c]*fx)*fy
                let expected=max(0,min(1,Double(ap[y*strideA+x*4+2-c])/255+d))
                XCTAssertEqual(Double(bp[y*strideB+x*4+2-c])/255,expected,accuracy:2.0/255,"GPU appearance decoder differs from CPU reference")
            }
        } }
        for y in stride(from:1600,to:2160,by:100) { for x in stride(from:0,to:900,by:100) {
            for c in 0..<4 { XCTAssertEqual(ap[y*strideA+x*4+c],bp[y*strideB+x*4+c],"Protected HUD differs from SR baseline") }
        } }
    }
    func testRejectUntrainedCoreMLProbe() throws {
        guard let path=ProcessInfo.processInfo.environment["NEURAL_POST_ANE_PROBE"] else { throw XCTSkip("Set NEURAL_POST_ANE_PROBE") }
        let device=try XCTUnwrap(MTLCreateSystemDefaultDevice())
        XCTAssertThrowsError(try ANEAppearance(url:URL(fileURLWithPath:path),device:device))
    }
}
