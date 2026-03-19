import onnx
import csv
from onnx import shape_inference

def export_onnx_to_csv(model_path, output_csv):
    # 1. Load the model and perform shape inference
    # This ensures intermediate layers have shape data
    model = onnx.load(model_path)
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    
    tensor_shapes = {}
    
    def extract_shapes(tensor_list):
        for tensor in tensor_list:
            name = tensor.name
            shape = []
            if tensor.type.tensor_type.HasField("shape"):
                for dim in tensor.type.tensor_type.shape.dim:
                    # dim_value > 0 is a fixed size; otherwise it's dynamic (None)
                    shape.append(dim.dim_value if dim.dim_value > 0 else "None")
            
            dtype = onnx.TensorProto.DataType.Name(tensor.type.tensor_type.elem_type)
            tensor_shapes[name] = {"shape": shape, "dtype": dtype}

    # Initialize shapes mapping
    extract_shapes(graph.input)
    extract_shapes(graph.output)
    extract_shapes(graph.value_info)

    # 2. Prepare data for CSV
    rows = []
    for node in graph.node:
        # Format Inputs
        input_list = []
        for inp in node.input:
            details = tensor_shapes.get(inp, {"shape": "Unknown", "dtype": "?"})
            input_list.append(f"{inp}{details['shape']}")
        
        # Format Outputs
        output_list = []
        for out in node.output:
            details = tensor_shapes.get(out, {"shape": "Unknown", "dtype": "?"})
            output_list.append(f"{out}{details['shape']}")

        rows.append({
            "Layer Name": node.name,
            "Op Type": node.op_type,
            "Inputs": ", ".join(input_list),
            "Outputs": ", ".join(output_list)
        })

    # 3. Write to CSV
    fieldnames = ["Layer Name", "Op Type", "Inputs", "Outputs"]
    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Successfully exported model details to: {output_csv}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    input_model = "./../tao_weight/lprnet_with_shapes.onnx"
    output_file = "model_layers.csv"
    export_onnx_to_csv(input_model, output_file)