import Foundation
import DLSSCore

/// Settings persisted by OBS. Defaults preserve the validated cascade controls.
package struct NRConfiguration: Codable, Equatable {
 var mode = "31then71-480"
 var enabled = true
 var upscaling = true
 var style = 0
 var intensity:Float = 1
 var layerIntensity:Float?
 enum IntensityMode: String, Codable { case final, layer }
 var intensityMode:IntensityMode?
 var finalGain:Float { intensityMode == .layer ? 1 : intensity }
 var layerGain:Float { intensityMode == .layer ? (layerIntensity ?? 1) : 1 }
 var appliesNR:Bool { enabled && (intensityMode == .layer ? layerGain : finalGain) > 0 }
 var tone:Float = 1
 var structure:Float = 1
 var automaticMask = false
 var skinStructure:Float = -1
 var maskStructure:Float = -1
 var noiseFrame = 0
 var model71 = ""
 var model31 = ""
 var srPath = ""
 var diffuseWhite:Float?
 var hdrStrength:Float?
 var colorStrength:Float?
 var uiCorrection:Bool?
 var uiX:Float?
 var uiY:Float?
 var uiWidth:Float?
 var uiHeight:Float?
 var temporalNR:Bool?
 var motionScaleX:Float?
 var motionScaleY:Float?
 var captureGuided = false
 var frameGenerationBackend = "mlx"
 var frameGenerationFactor = 1
 var frameGenerationHeight = 360
 var frameGenerationPath = ""
 var automatic:NeuralRenderingAutomaticMaskConfiguration? {
  automaticMask ? NeuralRenderingAutomaticMaskConfiguration(skinStructureStrength:skinStructure,automaticMaskStructureStrength:maskStructure) : nil
 }
 var title:String {
  mode == "71x2-480" ? "DLSS Fidelity" : mode == "31-480" ? "DLSS Performance" : "DLSS Balanced"
 }
 package func checked() throws -> NRConfiguration {
  guard ["mlx","apple"].contains(frameGenerationBackend),(1...6).contains(frameGenerationFactor), [360,540,720].contains(frameGenerationHeight), ["71x2-480","31then71-480","31-480"].contains(mode),
   (0...128).contains(style),(0...65535).contains(noiseFrame),
   (80...400).contains(diffuseWhite ?? 80),
   (0...2).contains(hdrStrength ?? 1),(0...2).contains(colorStrength ?? 1),
   [uiX,uiY,uiWidth,uiHeight].compactMap({$0}).allSatisfy({(0...100).contains($0)}),
   (-2...2).contains(motionScaleX ?? 1),(-2...2).contains(motionScaleY ?? 1),
   intensity.isFinite,(0...2).contains(intensity),(0...2).contains(layerIntensity ?? 1),tone.isFinite,(0...2).contains(tone),
   structure.isFinite,(0...2).contains(structure),skinStructure.isFinite,(-1...2).contains(skinStructure),
   maskStructure.isFinite,(-1...2).contains(maskStructure) else {
    throw NSError(domain:"DLSS NR",code:1,userInfo:[NSLocalizedDescriptionKey:"Invalid mode or model control range"])
   }
  return self
 }
}
