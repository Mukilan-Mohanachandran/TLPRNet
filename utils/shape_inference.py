from onnx import shape_inference
import onnx

# Load, infer shapes, and save/overwrite
model = onnx.load("./../tao_weight/us_lprnet_baseline18_deployable.onnx")
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, "./../tao_weight/lprnet_with_shapes.onnx")