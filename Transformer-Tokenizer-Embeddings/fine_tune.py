import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from datasets import load_dataset, Dataset as HFDataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QADataset(Dataset):
    """
    Custom dataset class for Question Answering with BERT.
    
    BERT Prompt Format for Question Answering:
    ==========================================
    
    BERT uses a specific input format for QA tasks:
    [CLS] question [SEP] context [SEP]
    
    Where:
    - [CLS]: Classification token at the beginning
    - [SEP]: Separator token between question and context
    - question: The question to be answered
    - context: The passage containing the answer
    
    Example:
    Input: "What is machine learning?" + "Machine learning is a subset of AI..."
    Tokenized: [CLS] What is machine learning ? [SEP] Machine learning is a subset of AI ... [SEP]
    
    The model predicts:
    - start_position: Token index where the answer begins
    - end_position: Token index where the answer ends
    
    Token IDs are then decoded back to text to extract the answer span.
    """
    
    def __init__(self, questions: List[str], contexts: List[str], answers: List[Dict], tokenizer, max_length: int = 512):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.contexts[idx]
        answer = self.answers[idx]
        
        # Tokenize using BERT's QA format: [CLS] question [SEP] context [SEP]
        # The tokenizer automatically handles special tokens
        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Find answer positions
        answer_text = answer["text"]
        start_char = answer["answer_start"]
        end_char = start_char + len(answer_text)
        
        # Convert character positions to token positions
        start_token = encoding.char_to_token(start_char)
        end_token = encoding.char_to_token(end_char - 1)
        
        # Handle cases where answer is truncated
        if start_token is None:
            start_token = 0
        if end_token is None:
            end_token = 0
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "start_positions": torch.tensor(start_token, dtype=torch.long),
            "end_positions": torch.tensor(end_token, dtype=torch.long)
        }


class BertQAFineTuner:
    """Fine-tuner for BERT Question Answering models."""
    
    def __init__(self, model_name: str = "distilbert-base-cased", use_quantization: bool = True):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
    def load_model_and_tokenizer(self):
        """Load the pre-trained model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Configure quantization if enabled
        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            self.model.to(self.device)
        
        logger.info("Model and tokenizer loaded successfully")
        return self.model, self.tokenizer
    
    def prepare_peft_model(self, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1):
        """Prepare model for PEFT fine-tuning."""
        logger.info("Preparing PEFT model with LoRA")
        
        # Prepare model for k-bit training
        if self.use_quantization:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["query", "value", "key", "dense"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="QUESTION_ANS",
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def load_squad_dataset(self, subset_size: Optional[int] = 1000):
        """Load and prepare SQuAD dataset."""
        logger.info("Loading SQuAD dataset")
        
        # Load SQuAD dataset
        dataset = load_dataset("squad", split="train")
        
        # Use subset for limited resources
        if subset_size and subset_size < len(dataset):
            dataset = dataset.select(range(subset_size))
            logger.info(f"Using subset of {subset_size} examples")
        
        # Extract data
        questions = []
        contexts = []
        answers = []
        
        for example in dataset:
            questions.append(example["question"])
            contexts.append(example["context"])
            # Take the first answer
            answers.append({
                "text": example["answers"]["text"][0],
                "answer_start": example["answers"]["answer_start"][0]
            })
        
        return questions, contexts, answers
    
    def prepare_dataset(self, questions: List[str], contexts: List[str], answers: List[Dict], 
                       train_split: float = 0.8):
        """Prepare training and validation datasets."""
        # Split data
        train_questions, val_questions, train_contexts, val_contexts, train_answers, val_answers = train_test_split(
            questions, contexts, answers, train_size=train_split, random_state=42
        )
        
        # Create datasets
        train_dataset = QADataset(train_questions, train_contexts, train_answers, self.tokenizer)
        val_dataset = QADataset(val_questions, val_contexts, val_answers, self.tokenizer)
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def create_training_arguments(self, output_dir: str, num_epochs: int = 2, 
                                 batch_size: int = 4, learning_rate: float = 2e-5,
                                 max_steps: int = 1000):
        """Create training arguments for limited resources."""
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Simulate larger batch size
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=100,
            max_steps=max_steps,  # Limit training steps
            fp16=True,  # Mixed precision training
            dataloader_num_workers=2,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb logging
        )
    
    def fine_tune(self, train_dataset, val_dataset, training_args):
        """Fine-tune the model."""
        logger.info("Starting fine-tuning")
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        logger.info(f"Model saved to {training_args.output_dir}")
        
        return trainer, eval_results


def main():
    """Main fine-tuning script."""
    # Configuration
    MODEL_NAME = "distilbert-base-cased"
    OUTPUT_DIR = "./models/fine_tuned_bert_qa"
    SUBSET_SIZE = 1000  # Use small subset for limited resources
    NUM_EPOCHS = 2
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    MAX_STEPS = 500  # Limit training steps
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize fine-tuner
    fine_tuner = BertQAFineTuner(model_name=MODEL_NAME, use_quantization=True)
    
    # Load model and tokenizer
    model, tokenizer = fine_tuner.load_model_and_tokenizer()
    
    # Prepare PEFT model
    model = fine_tuner.prepare_peft_model()
    
    # Load dataset
    questions, contexts, answers = fine_tuner.load_squad_dataset(subset_size=SUBSET_SIZE)
    
    # Prepare datasets
    train_dataset, val_dataset = fine_tuner.prepare_dataset(questions, contexts, answers)
    
    # Create training arguments
    training_args = fine_tuner.create_training_arguments(
        output_dir=OUTPUT_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS
    )
    
    # Fine-tune
    trainer, eval_results = fine_tuner.fine_tune(train_dataset, val_dataset, training_args)
    
    print("Fine-tuning completed!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Final evaluation results: {eval_results}")


if __name__ == "__main__":
    main()
