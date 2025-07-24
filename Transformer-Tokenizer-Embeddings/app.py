from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import os
import sys
import logging
from typing import Dict, Any

try:
    from bert_qa import QuantizedBertQA
except ImportError:
    # Fallback for development
    print("Warning: Could not import QuantizedBertQA. Using mock implementation.")
    
    class QuantizedBertQA:
        def __init__(self):
            pass
        def load_quantized_model(self, use_4bit=True):
            return None, None
        def answer_question(self, question, context):
            return {
                "answer": f"Mock answer for: {question}",
                "confidence": 0.95,
                "start_position": 0,
                "end_position": 10
            }

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
api = Api(app)

# Global model instance
bert_qa_model = None

def initialize_model():
    """Initialize the BERT QA model."""
    global bert_qa_model
    try:
        bert_qa_model = QuantizedBertQA()
        bert_qa_model.load_quantized_model(use_4bit=True)
        logger.info("BERT QA model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        bert_qa_model = QuantizedBertQA()  # Use mock version


class HealthCheck(Resource):
    """Health check endpoint."""
    
    def get(self):
        return {
            "status": "healthy",
            "service": "BERT Question Answering API",
            "model_loaded": bert_qa_model is not None
        }


class ModelInfo(Resource):
    """Model information endpoint."""
    
    def get(self):
        if bert_qa_model:
            return {
                "model_name": getattr(bert_qa_model, 'model_name', 'Unknown'),
                "model_type": "Quantized BERT for Question Answering",
                "framework": "PyTorch + Hugging Face Transformers",
                "quantization": "4-bit or 8-bit",
                "peft_enabled": True
            }
        else:
            return {"error": "Model not loaded"}, 500


class QuestionAnswering(Resource):
    """Main question answering endpoint."""
    
    def post(self):
        """
        Handle POST requests for question answering.
        
        Expected JSON format:
        {
            "question": "What is machine learning?",
            "context": "Machine learning is a subset of AI..."
        }
        """
        try:
            # Validate request data
            if not request.is_json:
                return {"error": "Request must be JSON"}, 400
            
            data = request.get_json()
            
            # Check required fields
            if "question" not in data or "context" not in data:
                return {
                    "error": "Missing required fields. Please provide 'question' and 'context'"
                }, 400
            
            question = data["question"].strip()
            context = data["context"].strip()
            
            # Validate input lengths
            if not question or not context:
                return {"error": "Question and context cannot be empty"}, 400
            
            if len(question) > 500:
                return {"error": "Question too long (max 500 characters)"}, 400
            
            if len(context) > 5000:
                return {"error": "Context too long (max 5000 characters)"}, 400
            
            # Get answer from model
            if bert_qa_model:
                result = bert_qa_model.answer_question(question, context)
                
                # Prepare response
                response = {
                    "question": question,
                    "answer": result["answer"],
                    "confidence": round(result["confidence"], 4),
                    "metadata": {
                        "start_position": result["start_position"],
                        "end_position": result["end_position"],
                        "context_length": len(context),
                        "question_length": len(question)
                    }
                }
                
                logger.info(f"Answered question: {question[:50]}...")
                return response
            else:
                return {"error": "Model not available"}, 503
                
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {"error": f"Internal server error: {str(e)}"}, 500


class BatchQuestionAnswering(Resource):
    """Batch question answering endpoint."""
    
    def post(self):
        """
        Handle batch question answering requests.
        
        Expected JSON format:
        {
            "qa_pairs": [
                {"question": "What is AI?", "context": "AI is..."},
                {"question": "What is ML?", "context": "ML is..."}
            ]
        }
        """
        try:
            if not request.is_json:
                return {"error": "Request must be JSON"}, 400
            
            data = request.get_json()
            
            if "qa_pairs" not in data:
                return {"error": "Missing 'qa_pairs' field"}, 400
            
            qa_pairs = data["qa_pairs"]
            
            if not isinstance(qa_pairs, list) or len(qa_pairs) == 0:
                return {"error": "qa_pairs must be a non-empty list"}, 400
            
            if len(qa_pairs) > 10:
                return {"error": "Maximum 10 QA pairs allowed per batch"}, 400
            
            # Validate each pair
            for i, pair in enumerate(qa_pairs):
                if not isinstance(pair, dict):
                    return {"error": f"QA pair {i} must be a dictionary"}, 400
                if "question" not in pair or "context" not in pair:
                    return {"error": f"QA pair {i} missing required fields"}, 400
            
            # Process batch
            if bert_qa_model:
                results = bert_qa_model.batch_answer(qa_pairs)
                
                response = {
                    "batch_size": len(qa_pairs),
                    "results": results
                }
                
                logger.info(f"Processed batch of {len(qa_pairs)} questions")
                return response
            else:
                return {"error": "Model not available"}, 503
                
        except Exception as e:
            logger.error(f"Error processing batch request: {e}")
            return {"error": f"Internal server error: {str(e)}"}, 500


# Register API routes
api.add_resource(HealthCheck, '/api/health')
api.add_resource(ModelInfo, '/api/model/info')
api.add_resource(QuestionAnswering, '/api/qa')
api.add_resource(BatchQuestionAnswering, '/api/qa/batch')


@app.route('/')
def index():
    """Root endpoint with API documentation."""
    return {
        "service": "BERT Question Answering API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This documentation",
            "GET /api/health": "Health check",
            "GET /api/model/info": "Model information",
            "POST /api/qa": "Single question answering",
            "POST /api/qa/batch": "Batch question answering"
        },
        "example_request": {
            "url": "/api/qa",
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "body": {
                "question": "What is machine learning?",
                "context": "Machine learning is a subset of artificial intelligence..."
            }
        },
        "curl_example": """
        curl -X POST http://localhost:5000/api/qa \\
          -H "Content-Type: application/json" \\
          -d '{
            "question": "What is machine learning?",
            "context": "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
          }'
        """
    }


if __name__ == '__main__':
    # Initialize model on startup
    initialize_model()
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting BERT QA API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
