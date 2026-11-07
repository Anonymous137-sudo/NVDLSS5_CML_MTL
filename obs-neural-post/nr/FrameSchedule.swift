import Foundation

/// Shared source-time playback clock. The small jitter reserve keeps inference
/// completion jitter out of phase spacing; all consumers use the same mapping.
struct FrameSchedule<Element> {
 struct Slot {let due:Double,value:Element}
 private(set) var slots:[Slot]=[]
 private(set) var selected:Element?
 private var clockOffset:Double?,lastSourceEnd:Double?
 private(set) var debtResets=0,expiredBeforeSelection=0,maxDepth=0,underruns=0
 var clockDelayMS:Double {(clockOffset ?? 0)*1000}
 mutating func clear() {slots=[];selected=nil;clockOffset=nil;lastSourceEnd=nil}
 mutating func sample(at time:Double)->Element? {
  var eligible=0
  while let first=slots.first,first.due<=time {selected=first.value;slots.removeFirst();eligible+=1}
  expiredBeforeSelection+=max(0,eligible-1)
  return selected
 }
 /// Predict a consumer's next presentation deadline without advancing another
 /// consumer's wall-clock selection into the future.
 func peek(at time:Double)->Element? {slots.last(where:{$0.due<=time})?.value ?? selected}
 mutating func enqueue(_ elements:[Element],at now:Double,interval:Double,reset:Bool,sourceEnd:Double?=nil,sourceTimes:[Double]?=nil) {
  guard !elements.isEmpty else{return}
  if reset {clear()}
  _ = sample(at:now)
  let end=sourceEnd ?? now
  if reset || interval<=0 || !interval.isFinite || !end.isFinite {
   slots=[Slot(due:now,value:elements.last!)];clockOffset=nil;lastSourceEnd=end;maxDepth=max(maxDepth,1);return
  }
  let step=max(1.0/240,interval/Double(elements.count))
  let reserve=max(1.0/60,interval*0.25)
  let explicit=sourceTimes.flatMap{times in times.count==elements.count && times.allSatisfy({$0.isFinite}) && zip(times,times.dropFirst()).allSatisfy({$0<$1}) ? times : nil}
  let firstSource=explicit?.first ?? end-interval+step
  let age=now-end
  if clockOffset==nil {clockOffset=now+reserve-firstSource}
  var firstDue=firstSource+clockOffset!
  if firstDue<now {
   let resumed=max(now+min(step,1.0/120),(slots.last?.due ?? now)+step)
   clockOffset!+=resumed-firstDue;firstDue=resumed;underruns+=1
  }
  if firstDue-now>0.20 || slots.count+elements.count>12 || clockOffset!>age+interval+0.20 || (lastSourceEnd.map{end<=$0} ?? false) {
   slots=[];clockOffset=now+reserve-firstSource;firstDue=now+reserve;debtResets+=1
  }
  for (index,value) in elements.enumerated() {slots.append(Slot(due:firstDue+(explicit.map{$0[index]-firstSource} ?? Double(index)*step),value:value))}
  lastSourceEnd=end;maxDepth=max(maxDepth,slots.count)
 }
}
