import os

import torch
from transformers import AutoTokenizer, AutoModel


def export_embedder():
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Downloading {model_id} from HuggingFace...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    # Create dummy inputs for ONNX tracing
    # MiniLM uses input_ids, attention_mask, and token_type_ids
    dummy_text = ["This is a test sentence for ONNX export."]
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)

    onnx_path = "embedder.onnx"
    print("Exporting model to ONNX FP32 format...")

    # Export to ONNX
    # We specify dynamic axes so the model can accept variable batch sizes and sequence lengths
    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']),
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'token_type_ids': {0: 'batch_size', 1: 'sequence_length'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
            'pooler_output': {0: 'batch_size'}
        }
    )
    print(f"✅ Exported successfully to {onnx_path}")

    # Quantize to INT8 for extreme lightweight deployment
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quant_path = "embedder_quant.onnx"
        print("Applying INT8 dynamic quantization...")
        quantize_dynamic(
            model_input=onnx_path,
            model_output=quant_path,
            weight_type=QuantType.QUInt8
        )
        print(f"✅ Quantized model saved to {quant_path}")

        # We also need to save the tokenizer configurations locally
        # so our deployment environment doesn't need to connect to HuggingFace
        os.makedirs("tokenizer_data", exist_ok=True)
        tokenizer.save_pretrained("tokenizer_data")
        print("✅ Tokenizer files saved to 'tokenizer_data/'")

    except ImportError:
        print("onnxruntime not installed. Skipping quantization.")


if __name__ == "__main__":
    os.makedirs("src", exist_ok=True)  # Ensure running from root works
    export_embedder()
