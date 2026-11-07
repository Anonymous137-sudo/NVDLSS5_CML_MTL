import Foundation
import MLX
@main struct DifferenceRegression {
 static func main() {
  let a=MLXArray.ones([1,360,640,3],dtype:.float16)*Float(0.7)
  let b=MLXArray.ones([1,360,640,3],dtype:.float16)*Float(0.5)
  let old=mean(abs(a-b)).item(Float.self)
  let value=mean(abs(a.asType(.float32)-b.asType(.float32))).item(Float.self)
  precondition(value.isFinite && abs(value-0.2)<0.001)
  precondition(JSONSerialization.isValidJSONObject(["difference":value]))
  precondition(!JSONSerialization.isValidJSONObject(["difference":Double.infinity]))
  print("PASS: normalized high-difference proxy remains finite and serializable; old half mean=\(old), float32 mean=\(value)")
 }
}
