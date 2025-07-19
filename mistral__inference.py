from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

import os 
import torch
import safetensors.torch

if __name__ == "__main__":

    # Create paths to model files
    tokenizer_path = os.path.join(os.getcwd(), "mistral_models", "7B-v0.3", "tokenizer.model.v3")
    model_path = os.path.join(os.getcwd(), "mistral_models", "7B-v0.3")
    lora_path = os.path.join(os.getcwd(), "mistral_models", "test_ultra", "checkpoints", "checkpoint_000100", "consolidated", "lora.safetensors")

    print(f"Tokenizer path: {tokenizer_path}")
    print(f"Model path: {model_path}")
    print(f"Lora path: {lora_path}")

    # Load tokenizer
    tokenizer = MistralTokenizer.from_file(tokenizer_path)

    # Load model
    model = Transformer.from_folder(model_path)
    
    # Load and convert LoRA weights to bfloat16
    print("Loading LoRA weights and converting to bfloat16...")
    lora_state_dict = safetensors.torch.load_file(lora_path)
    
    # Convert all tensors to bfloat16
    converted_state_dict = {}
    for key, tensor in lora_state_dict.items():
        converted_state_dict[key] = tensor.to(torch.bfloat16)
    
    # Save converted weights temporarily
    temp_lora_path = lora_path.replace('.safetensors', '_bfloat16.safetensors')
    safetensors.torch.save_file(converted_state_dict, temp_lora_path)
    
    # Load the converted LoRA weights
    model.load_lora(temp_lora_path)
    
    # Clean up temporary file
    os.remove(temp_lora_path)

    # Create completion request
    completion_request = ChatCompletionRequest(messages=[UserMessage(content="¿Se viola el derecho a la salud cuando las entidades prestadoras de servicios (EPS) imponen obstáculos para el acceso, como la exigencia de trámites burocráticos complejos o la negación de servicios no incluidos en el Plan Obligatorio de Salud (POS)?")])

    # Generate tokens
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    out_tokens, _ = generate([tokens], model, max_tokens=512, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

    print(result)

    