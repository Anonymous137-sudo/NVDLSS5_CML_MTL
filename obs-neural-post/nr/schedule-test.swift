import Foundation
@main struct ScheduleTests {
 static func main() {
  var s=FrameSchedule<Int>()
  s.enqueue([0],at:0,interval:0,reset:true,sourceEnd:0)
  precondition(s.sample(at:0)==0)
  s.enqueue(Array(1...6),at:0.12,interval:0.1,reset:false,sourceEnd:0.1)
  let due=s.slots.map(\.due)
  precondition(abs(due[0]-0.145)<1e-9)
  for i in 1..<6 {precondition(abs(due[i]-due[i-1]-1.0/60)<1e-9)}
  precondition(s.sample(at:0.14)==0)
  // Deadline sampling must not move another consumer's wall-clock result.
  precondition(s.peek(at:0.146)==1 && s.sample(at:0.14)==0)
  precondition(s.sample(at:0.146)==1 && s.sample(at:0.146)==1)
  // A modest arrival-time perturbation must not shift the source-time phases.
  s.enqueue(Array(7...12),at:0.226,interval:0.1,reset:false,sourceEnd:0.2)
  precondition(abs(s.slots.suffix(6).first!.due-due[0]-0.1)<1e-9 && s.underruns==0)
  precondition(s.sample(at:0.229)==6 && s.sample(at:0.246)==7)
  for i in 0..<20 {s.enqueue(Array((20+i*6)..<(26+i*6)),at:0.3,interval:0.1,reset:false,sourceEnd:0.3+Double(i)*0.001)}
  precondition(s.slots.count<=12 && s.maxDepth<=12 && s.debtResets>0)
  var explicit=FrameSchedule<Int>()
  explicit.enqueue([1,2,3],at:0.20,interval:0.1,reset:false,sourceEnd:0.1,sourceTimes:[1.0/60,4.0/60,0.1])
  precondition(abs(explicit.slots[1].due-explicit.slots[0].due-0.05)<1e-9)
  precondition(abs(explicit.slots[2].due-explicit.slots[1].due-1.0/30)<1e-9)
  var late=FrameSchedule<Int>()
  late.enqueue(Array(1...6),at:0.12,interval:0.1,reset:false,sourceEnd:0.1)
  late.enqueue(Array(7...12),at:0.30,interval:0.1,reset:false,sourceEnd:0.2)
  precondition(late.underruns==1 && late.slots.first!.due<=0.30+1.0/60+1e-9,"Late work waited for another full jitter reserve")
  s.enqueue([999],at:2,interval:0,reset:true,sourceEnd:2)
  precondition(s.sample(at:2)==999 && s.slots.isEmpty)
  s.clear();precondition(s.sample(at:3)==nil && s.peek(at:4)==nil)
  print("PASS: six-phase spacing, producer-jitter isolation, independent deadline consumers, bounded queue, reset")
 }
}
