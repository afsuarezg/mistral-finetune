#!/usr/bin/env python3
"""
Continued Pretraining for Mistral Models
Supports both full fine-tuning and LoRA approaches
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
import datasets
from datasets import load_dataset
import wandb
from typing import Dict, List, Optional
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Custom dataset for continued pretraining on raw text"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize all texts
        logger.info(f"Tokenizing {len(texts)} texts...")
        self.examples = []
        
        for text in texts:
            # Tokenize with truncation
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None
            )
            
            # Only keep examples with reasonable length
            if len(encoded['input_ids']) > 50:  # Minimum length threshold
                self.examples.append(encoded['input_ids'])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)

class MistralContinuedPretrainer:
    """Main class for continued pretraining of Mistral models"""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        use_lora: bool = True,
        max_length: int = 2048,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.use_lora = use_lora
        self.max_length = max_length
        self.device = device
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer for {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        logger.info(f"Loading model {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Use bf16 for efficiency
            device_map=device,
            trust_remote_code=True
        )
        
        # Setup LoRA if requested
        if use_lora:
            self.setup_lora()
    
    def setup_lora(self):
        """Configure LoRA for parameter-efficient training"""
        logger.info("Setting up LoRA configuration")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # Rank
            lora_alpha=32,  # Scaling parameter
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "v_proj", 
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def prepare_dataset(self, data_path: str, validation_split: float = 0.1):
        """Prepare dataset from text files or HuggingFace dataset"""
        
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            # Load from JSON/JSONL file
            with open(data_path, 'r') as f:
                if data_path.endswith('.jsonl'):
                    data = [json.loads(line)['text'] for line in f]
                else:
                    data = json.load(f)
                    if isinstance(data, dict) and 'texts' in data:
                        data = data['texts']
                    elif isinstance(data, list) and isinstance(data[0], dict):
                        data = [item['text'] for item in data]
        
        elif data_path.endswith('.txt'):
            # Load from text file
            with open(data_path, 'r') as f:
                text = f.read()
            # Split into chunks (simple approach)
            chunk_size = self.max_length * 3  # Rough estimate
            data = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        else:
            # Try loading as HuggingFace dataset
            try:
                dataset = load_dataset(data_path)
                data = dataset['train']['text']
            except Exception as e:
                raise ValueError(f"Could not load data from {data_path}: {e}")
        
        # Split into train/validation
        split_idx = int(len(data) * (1 - validation_split))
        train_texts = data[:split_idx]
        val_texts = data[split_idx:]
        
        logger.info(f"Prepared {len(train_texts)} training and {len(val_texts)} validation examples")
        
        # Create datasets
        train_dataset = TextDataset(train_texts, self.tokenizer, self.max_length)
        val_dataset = TextDataset(val_texts, self.tokenizer, self.max_length)
        
        return train_dataset, val_dataset
    
    def train(
        self,
        train_dataset,
        val_dataset,
        output_dir: str = "./mistral-continued",
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
        max_grad_norm: float = 1.0,
        use_wandb: bool = False
    ):
        """Train the model with specified parameters"""
        
        if use_wandb:
            wandb.init(project="mistral-continued-pretraining")
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8  # For efficiency
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=True,
            bf16=True,  # Use bfloat16 for training
            remove_unused_columns=False,
            report_to="wandb" if use_wandb else None,
            run_name=f"mistral-continued-{num_epochs}epochs" if use_wandb else None,
            max_grad_norm=max_grad_norm,
            prediction_loss_only=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        
        if use_wandb:
            wandb.finish()
        
        return trainer

# Alternative approach using native PyTorch (for more control)
class MistralTrainerPyTorch:
    """PyTorch-native training loop for more control"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def train_epoch(self, dataloader, optimizer, scheduler, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for step, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch.to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if step % 100 == 0:
                logger.info(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        
        return total_loss / len(dataloader)

# Example usage and configuration
def main():
    """Example usage of the continued pretraining system"""
    
    # Configuration
    config = {
        "model_name": "mistralai/Mistral-7B-v0.1",
        "data_path": "your_dataset.jsonl",  # Replace with your data
        "output_dir": "./mistral-continued-output",
        "use_lora": True,
        "max_length": 2048,
        "num_epochs": 3,
        "batch_size": 2,  # Adjust based on your GPU memory
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
        "use_wandb": False,
    }
    
    # Initialize trainer
    trainer = MistralContinuedPretrainer(
        model_name=config["model_name"],
        use_lora=config["use_lora"],
        max_length=config["max_length"]
    )
    
    # Prepare data (you need to provide your own dataset)
    # train_dataset, val_dataset = trainer.prepare_dataset(config["data_path"])
    
    # For demo purposes, create dummy data
    dummy_texts = [
        "This is example text for continued pretraining. " * 50,
        "Another example with different content for training. " * 50,
        "More training data to help the model learn. " * 50,
    ] * 100  # Repeat to create more examples
    
    from sklearn.model_selection import train_test_split
    train_texts, val_texts = train_test_split(dummy_texts, test_size=0.1)
    
    train_dataset = TextDataset(train_texts, trainer.tokenizer, config["max_length"])
    val_dataset = TextDataset(val_texts, trainer.tokenizer, config["max_length"])
    
    # Start training
    trained_model = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=config["output_dir"],
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        use_wandb=config["use_wandb"]
    )
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
