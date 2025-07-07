#!/usr/bin/env python3
"""
Mistral 7B Fine-tuning Script
Extracted from Own_mistral_finetune_7b.ipynb

This script demonstrates how to fine-tune Mistral 7B using LoRA.
"""

import os
import json
import platform
import pandas as pd
import yaml
import shutil
import pprint
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


def setup_environment():
    """Setup environment variables and authentication."""
    print(f"Platform: {platform.system()}")
    
    # Load environment variables
    load_dotenv()
    huggingface_key = os.getenv('huggingface_key')
    
    # Login to Hugging Face
    if huggingface_key:
        login(token=huggingface_key)
        print("Logged in to Hugging Face")
    else:
        print("Warning: No Hugging Face key found in environment variables")
    
    # Set CUDA environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    print("Environment setup complete")


def download_model():
    """Download Mistral 7B model."""
    print("Setting up model directory...")
    
    # Change to project directory
    os.chdir(r'C:\Users\Andres.DESKTOP-D77KM25\Documents\Legal_tech_projects\Mistral-Finetune\mistral-finetune')
    print(f"Current directory: {os.getcwd()}")
    
    # Create model directory
    mistral_models_path = Path.cwd().joinpath('mistral_models', '7B-v0.3')
    mistral_models_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Model will be downloaded to: {mistral_models_path}")
    
    # Download model files
    print("Downloading Mistral 7B model...")
    snapshot_download(
        repo_id="mistralai/Mistral-7B-v0.3", 
        allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], 
        local_dir=mistral_models_path
    )
    
    print("Model download complete")
    return mistral_models_path


def prepare_dataset():
    """Prepare training dataset from UltraChat."""
    print("Preparing dataset...")
    
    # Navigate to data directory
    data_dir = Path.cwd().joinpath('data')
    data_dir.mkdir(exist_ok=True)
    os.chdir(data_dir)
    
    # Load UltraChat dataset
    print("Loading UltraChat dataset...")
    df = pd.read_parquet('https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/resolve/main/data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet')
    
    # Split into train and eval
    df_train = df.sample(frac=0.95, random_state=200)
    df_eval = df.drop(df_train.index)
    
    print(f"Training samples: {len(df_train)}")
    print(f"Evaluation samples: {len(df_eval)}")
    
    # Save to JSONL format
    df_train.to_json("ultrachat_chunk_train.jsonl", orient="records", lines=True)
    df_eval.to_json("ultrachat_chunk_eval.jsonl", orient="records", lines=True)
    
    print("Dataset saved to JSONL format")
    
    # Return to project directory
    os.chdir(Path.cwd().parent)
    
    return data_dir


def reformat_data(data_dir):
    """Reformat data to correct format for training."""
    print("Reformatting data...")
    
    train_file = data_dir / "ultrachat_chunk_train.jsonl"
    eval_file = data_dir / "ultrachat_chunk_eval.jsonl"
    
    # Reformat training data
    print("Reformatting training data...")
    os.system(f'python -m utils.reformat_data "{train_file}"')
    
    # Reformat evaluation data
    print("Reformatting evaluation data...")
    os.system(f'python -m utils.reformat_data "{eval_file}"')
    
    # Validate the reformatted data
    print("Validating reformatted data...")
    os.system('python -m utils.validate_data --train_yaml example/7B.yaml')

    print("Data reformatting complete")





def create_training_config(data_dir, model_path):
    """Create training configuration YAML file."""
    print("Creating training configuration...")
    
    config = {
        "data": {
            "instruct_data": str(data_dir / "ultrachat_chunk_train.jsonl"),
            "data": None,  #"/content/drive/My Drive/Mistral/Data/output.jsonl",  # Optional pretraining data
            "eval_instruct_data": str(data_dir / "ultrachat_chunk_eval.jsonl")
        },
        "model_id_or_path": str(model_path),
        "lora": {
            "rank": 64
        },
        "seq_len": 8192,
        "batch_size": 1,
        "num_microbatches": 8,
        "max_steps": 100,
        "optim": {
            "lr": 1e-5,
            "weight_decay": 0.1,
            "pct_start": 0.05
        },
        "seed": 0,
        "log_freq": 1,
        "eval_freq": 200,
        "no_eval": False,
        "ckpt_freq": 100,
        "save_adapters": True,
        "run_dir": str(Path.cwd() / "mistral_models" / "test_ultra")
    }
    
    # Pretty print the configuration
    print("Configuration:")
    pprint.pprint(config, indent=2, width=80)
    
    # Save configuration
    with open('example.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    print("Training configuration saved to example.yaml")
    return config


def start_training():
    """Start the training process."""
    print("Starting training...")
    
    # Check if we're on Windows
    if platform.system() == "Windows":
        print("Windows detected - using alternative training approach...")
        
        # Set environment variables to simulate distributed training
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["TORCHELASTIC_RESTART_COUNT"] = "0"
        
        # Try direct Python execution instead of torchrun
        try:
            print("Attempting direct Python execution...")
            os.system('python -m train example.yaml')
        except Exception as e:
            print(f"Direct execution failed: {e}")
            print("\n=== WINDOWS TRAINING LIMITATIONS ===")
            print("PyTorch distributed training has known issues on Windows.")
            print("\nRecommended solutions:")
            print("1. Use WSL2 (Windows Subsystem for Linux):")
            print("   - Install WSL2 from Microsoft Store")
            print("   - Install Ubuntu on WSL2")
            print("   - Run your training there")
            print("\n2. Use Google Colab (free):")
            print("   - Upload your code to Google Colab")
            print("   - Run training in the cloud")
            print("\n3. Use a cloud service (AWS, Azure, etc.)")
            print("\n4. Manual command (if you want to try):")
            print("   torchrun --nproc-per-node=1 --master-port=29500 -m train example.yaml")
    else:
        # For non-Windows systems, use normal torchrun
        try:
            os.system('torchrun --nproc-per-node=1 -m train example.yaml')
        except Exception as e:
            print(f"Training failed: {e}")


def save_model_to_drive():
    """Save the trained model to Google Drive (if available)."""
    print("Saving model to Google Drive...")
    
    folder_to_save = '/content/test_ultra'
    drive_destination_path = '/content/drive/My Drive/Mistral/Saved_Models/test_ultra_saved'
    
    try:
        shutil.copytree(folder_to_save, drive_destination_path)
        print(f"Folder '{folder_to_save}' successfully saved to '{drive_destination_path}'")
    except FileExistsError:
        print(f"Destination folder '{drive_destination_path}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")


def setup_inference():
    """Setup for model inference."""
    print("Setting up inference...")
    
    # Install mistral_inference if not already installed
    try:
        import mistral_inference
        print("mistral_inference already installed")
    except ImportError:
        print("Installing mistral_inference...")
        os.system('pip install mistral_inference')


def run_inference(model_path, lora_path=None):
    """Run inference with the trained model."""
    print("Running inference...")
    
    # Initialize tokenizer and model
    tokenizer = MistralTokenizer.from_file(str(model_path / "tokenizer.model.v3"))
    model = Transformer.from_folder(str(model_path))
    
    if lora_path:
        model.load_lora(str(lora_path))
        print("LoRA adapter loaded")
    
    # Test prompts
    test_prompts = [
        "Del análisis del trámite legislativo realizado por la Corte Constitucional, se concluye que el Proyecto de Ley No. 154 de 2012 Senado,",
        "la doble tributación, conlleva la aplicación de dos o",
        "la doble tributación, conlleva la aplicación de dos o más normas tributarias"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")
        
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompt)]
        )
        
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        
        out_tokens, _ = generate(
            [tokens], 
            model, 
            max_tokens=64, 
            temperature=0.0, 
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id
        )
        
        result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        print(f"Response: {result}")


def read_jsonl_file(file_path):
    """Read and parse a JSONL file."""
    print(f"Reading JSONL file: {file_path}")
    
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    json_object = json.loads(line)
                    data.append(json_object)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line.strip()}")
                    print(f"Error details: {e}")
    
    print(f"Successfully read {len(data)} entries from {file_path}")
    if data:
        print("First 3 entries:")
        for i, entry in enumerate(data[:3]):
            print(f"Entry {i+1}: {str(entry)[:200]}...")
    
    return data


def main():
    """Main execution function."""
    print("=== Mistral 7B Fine-tuning Script ===")
    
    # Setup environment
    print("\n Setting up environment...")
    setup_environment()
    
    # Download model (uncomment if needed)
    print("\nDownloading model...")
    # model_path = download_model()
    model_path = Path.cwd().joinpath('mistral_models', '7B-v0.3')
    
    # Prepare dataset (uncomment if needed)
    # data_dir = prepare_dataset()
    print("\nPreparing dataset...")
    data_dir = Path.cwd().joinpath('data')
    
    # Reformat data (uncomment if needed)
    print("\nReformatting data...")
    # reformat_data(data_dir)

    # Create training configuration
    print("\nCreating training configuration...")
    config = create_training_config(data_dir, model_path)
    
    # Start training
    print("\nStarting training...")
    start_training()
    
    # Setup inference
    # print("\nSetting up inference...")
    # setup_inference()
    
    # Run inference (uncomment if needed)
    # lora_path = Path.cwd().joinpath('mistral_models', 'test_ultra', 'checkpoints', 'checkpoint_000100', 'consolidated', 'lora.safetensors')
    # run_inference(model_path, lora_path)
    
    print("Script execution complete!")


if __name__ == "__main__":
    main()
