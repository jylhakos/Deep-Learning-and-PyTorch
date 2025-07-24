import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
from typing import Dict, List, Optional


class QuantizedBertQA:
    """
    Quantized BERT model for Question Answering with PEFT fine-tuning support.
    """
    
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_quantized_model(self, use_4bit: bool = True):
        """
        Load a quantized BERT model for question answering.
        """
        print(f"Loading quantized model: {self.model_name}")
        
        # Configure quantization
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model with quantization
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        print(f"Model loaded on device: {self.device}")
        return self.model, self.tokenizer
    
    def prepare_peft_model(self, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1):
        """
        Prepare model for Parameter Efficient Fine-Tuning using LoRA.
        """
        # Prepare model for k-bit training
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
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def answer_question(self, question: str, context: str, max_length: int = 512) -> Dict:
        """
        Answer a question given the context.
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_quantized_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            question,
            context,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
        
        # Find the answer span
        start_idx = torch.argmax(start_logits, dim=1).item()
        end_idx = torch.argmax(end_logits, dim=1).item()
        
        # Ensure end comes after start
        if end_idx < start_idx:
            end_idx = start_idx
        
        # Extract answer tokens
        input_ids = inputs["input_ids"][0]
        answer_tokens = input_ids[start_idx:end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # Calculate confidence scores
        start_confidence = torch.softmax(start_logits, dim=1)[0][start_idx].item()
        end_confidence = torch.softmax(end_logits, dim=1)[0][end_idx].item()
        confidence = (start_confidence + end_confidence) / 2
        
        return {
            "answer": answer,
            "confidence": confidence,
            "start_position": start_idx,
            "end_position": end_idx
        }
    
    def batch_answer(self, qa_pairs: List[Dict[str, str]]) -> List[Dict]:
        """
        Answer multiple questions in batch.
        """
        results = []
        for pair in qa_pairs:
            result = self.answer_question(pair["question"], pair["context"])
            result["question"] = pair["question"]
            results.append(result)
        return results
    
    def save_model(self, output_dir: str):
        """
        Save the fine-tuned model.
        """
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"Model saved to {output_dir}")
        else:
            print("Model does not support save_pretrained method")
    
    def load_finetuned_model(self, model_path: str):
        """
        Load a previously fine-tuned model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.model.to(self.device)
        print(f"Fine-tuned model loaded from {model_path}")


# Example usage
if __name__ == "__main__":
    # Initialize the QA model
    bert_qa = QuantizedBertQA()
    
    # Load quantized model
    model, tokenizer = bert_qa.load_quantized_model(use_4bit=True)
    
    # Example question answering
    question = "What is machine learning?"
    context = """
    Machine learning is a subset of artificial intelligence (AI) that provides systems 
    the ability to automatically learn and improve from experience without being explicitly programmed. 
    Machine learning focuses on the development of computer programs that can access data 
    and use it to learn for themselves.
    """
    
    result = bert_qa.answer_question(question, context)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.4f}")
