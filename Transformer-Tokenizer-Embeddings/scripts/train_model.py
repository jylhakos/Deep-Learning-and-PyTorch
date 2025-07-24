#!/usr/bin/env python3
"""
Training script for BERT Question Answering model with quantization and PEFT.
This script demonstrates fine-tuning a quantized BERT model for QA tasks.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from fine_tune import BertQAFineTuner
    from bert_qa import QuantizedBertQA
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed and the virtual environment is activated.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune BERT for Question Answering")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-cased",
        help="Pre-trained model name from Hugging Face Hub"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/fine_tuned_bert_qa",
        help="Directory to save the fine-tuned model"
    )
    
    parser.add_argument(
        "--subset_size",
        type=int,
        default=1000,
        help="Number of training examples to use (for limited resources)"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum number of training steps"
    )
    
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        default=True,
        help="Use 4-bit quantization"
    )
    
    parser.add_argument(
        "--use_peft",
        action="store_true",
        default=True,
        help="Use Parameter Efficient Fine-Tuning (LoRA)"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()
    
    logger.info("=== BERT QA Fine-tuning Script ===")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Subset size: {args.subset_size}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Use quantization: {args.use_quantization}")
    logger.info(f"Use PEFT: {args.use_peft}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize fine-tuner
        logger.info("Initializing fine-tuner...")
        fine_tuner = BertQAFineTuner(
            model_name=args.model_name,
            use_quantization=args.use_quantization
        )
        
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model, tokenizer = fine_tuner.load_model_and_tokenizer()
        
        # Apply PEFT if enabled
        if args.use_peft:
            logger.info("Applying PEFT (LoRA)...")
            model = fine_tuner.prepare_peft_model()
        
        # Load dataset
        logger.info("Loading SQuAD dataset...")
        questions, contexts, answers = fine_tuner.load_squad_dataset(
            subset_size=args.subset_size
        )
        
        # Prepare datasets
        logger.info("Preparing training and validation datasets...")
        train_dataset, val_dataset = fine_tuner.prepare_dataset(
            questions, contexts, answers
        )
        
        # Create training arguments
        logger.info("Setting up training arguments...")
        training_args = fine_tuner.create_training_arguments(
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps
        )
        
        # Fine-tune the model
        logger.info("Starting fine-tuning...")
        trainer, eval_results = fine_tuner.fine_tune(
            train_dataset, val_dataset, training_args
        )
        
        # Test the model
        logger.info("Testing the fine-tuned model...")
        bert_qa = QuantizedBertQA(model_name=args.output_dir)
        bert_qa.load_finetuned_model(args.output_dir)
        
        # Test question
        test_question = "What is machine learning?"
        test_context = """
        Machine learning is a subset of artificial intelligence (AI) that provides systems 
        the ability to automatically learn and improve from experience without being explicitly programmed. 
        Machine learning focuses on the development of computer programs that can access data 
        and use it to learn for themselves.
        """
        
        result = bert_qa.answer_question(test_question, test_context)
        
        logger.info("=== Test Results ===")
        logger.info(f"Question: {test_question}")
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Confidence: {result['confidence']:.4f}")
        
        logger.info("=== Training Complete ===")
        logger.info(f"Model saved to: {args.output_dir}")
        logger.info(f"Final evaluation results: {eval_results}")
        
        # Save training summary
        summary_file = os.path.join(args.output_dir, "training_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("BERT QA Fine-tuning Summary\n")
            f.write("=" * 30 + "\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Training samples: {len(train_dataset)}\n")
            f.write(f"Validation samples: {len(val_dataset)}\n")
            f.write(f"Epochs: {args.num_epochs}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Learning rate: {args.learning_rate}\n")
            f.write(f"Max steps: {args.max_steps}\n")
            f.write(f"Quantization: {args.use_quantization}\n")
            f.write(f"PEFT: {args.use_peft}\n")
            f.write(f"Final evaluation: {eval_results}\n")
            f.write(f"Test question: {test_question}\n")
            f.write(f"Test answer: {result['answer']}\n")
            f.write(f"Test confidence: {result['confidence']:.4f}\n")
        
        logger.info(f"Training summary saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e


if __name__ == "__main__":
    main()
