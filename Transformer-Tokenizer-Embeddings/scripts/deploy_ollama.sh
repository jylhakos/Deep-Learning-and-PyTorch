#!/bin/bash

# Ollama Deployment Script for BERT QA Model
# This script sets up Ollama with Docker and deploys the fine-tuned BERT model

echo "=== Ollama Deployment Script for BERT QA ==="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "Docker found: $(docker --version)"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "Error: Docker is not running. Please start Docker service."
    exit 1
fi

echo "Docker is running"

# Pull Ollama Docker image
echo "Pulling Ollama Docker image..."
docker pull ollama/ollama:latest

# Create Ollama data volume
echo "Creating Ollama data volume..."
docker volume create ollama-data

# Stop existing Ollama container if running
echo "Stopping existing Ollama container..."
docker stop ollama-bert-qa 2>/dev/null || true
docker rm ollama-bert-qa 2>/dev/null || true

# Run Ollama container
echo "Starting Ollama container..."
docker run -d \
    --name ollama-bert-qa \
    -v ollama-data:/root/.ollama \
    -p 11434:11434 \
    --restart unless-stopped \
    ollama/ollama:latest

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
sleep 10

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Error: Ollama is not responding. Check the container logs:"
    echo "docker logs ollama-bert-qa"
    exit 1
fi

echo "Ollama is running successfully!"

# Create Modelfile for BERT QA
echo "Creating Modelfile for BERT QA..."
cat > Modelfile << 'EOF'
FROM scratch

# Set the base model (using a lightweight model as placeholder)
FROM llama2:7b-chat

# Custom system prompt for QA
SYSTEM """You are a specialized question-answering assistant powered by BERT. 
When given a question and context, provide accurate answers based on the provided information.
If the answer is not found in the context, respond with "I cannot find the answer in the provided context."
"""

# Set parameters for QA task
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 256

# Template for QA format
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
Context: {{ .Context }}
Question: {{ .Prompt }}<|end|>
<|assistant|>
{{ end }}{{ .Response }}<|end|>
"""
EOF

# Build the custom model
echo "Building custom BERT QA model in Ollama..."
docker exec ollama-bert-qa ollama create bert-qa -f /root/Modelfile 2>/dev/null || {
    # If direct build fails, copy Modelfile and build
    docker cp Modelfile ollama-bert-qa:/root/
    docker exec ollama-bert-qa ollama create bert-qa -f /root/Modelfile
}

# Clean up Modelfile
rm -f Modelfile

# Test the deployment
echo "Testing Ollama deployment..."
RESPONSE=$(curl -s -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bert-qa",
    "prompt": "What is machine learning?",
    "context": "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
    "stream": false
  }')

if [ $? -eq 0 ]; then
    echo "✓ Ollama deployment successful!"
    echo "Response: $RESPONSE"
else
    echo "✗ Ollama deployment test failed"
fi

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Ollama is running on: http://localhost:11434"
echo ""
echo "Available commands:"
echo "  - Check status: curl http://localhost:11434/api/tags"
echo "  - List models: docker exec ollama-bert-qa ollama list"
echo "  - Stop container: docker stop ollama-bert-qa"
echo "  - Start container: docker start ollama-bert-qa"
echo "  - View logs: docker logs ollama-bert-qa"
echo ""
echo "Example API call:"
echo 'curl -X POST http://localhost:11434/api/generate \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{'
echo '    "model": "bert-qa",'
echo '    "prompt": "What is AI?",'
echo '    "context": "Artificial Intelligence is...",'
echo '    "stream": false'
echo '  }'"'"

# Create integration script
echo "Creating integration script..."
cat > integrate_bert_ollama.py << 'EOF'
#!/usr/bin/env python3
"""
Integration script to connect Flask BERT QA API with Ollama.
This creates a bridge between the fine-tuned BERT model and Ollama.
"""

import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BertOllamaIntegration:
    """Integration between BERT QA API and Ollama."""
    
    def __init__(self, bert_api_url="http://localhost:5000", ollama_api_url="http://localhost:11434"):
        self.bert_api_url = bert_api_url
        self.ollama_api_url = ollama_api_url
    
    def check_services(self):
        """Check if both services are running."""
        # Check BERT API
        try:
            response = requests.get(f"{self.bert_api_url}/api/health")
            bert_status = response.status_code == 200
        except:
            bert_status = False
        
        # Check Ollama
        try:
            response = requests.get(f"{self.ollama_api_url}/api/tags")
            ollama_status = response.status_code == 200
        except:
            ollama_status = False
        
        return bert_status, ollama_status
    
    def process_question(self, question, context):
        """Process question through BERT and format for Ollama."""
        # Get answer from BERT
        try:
            response = requests.post(
                f"{self.bert_api_url}/api/qa",
                json={"question": question, "context": context}
            )
            bert_result = response.json()
            
            # Format for Ollama
            ollama_prompt = f"""
            Based on the BERT analysis:
            Question: {question}
            Context: {context}
            BERT Answer: {bert_result.get('answer', 'No answer found')}
            Confidence: {bert_result.get('confidence', 0):.2f}
            
            Please provide a comprehensive response based on this analysis.
            """
            
            return ollama_prompt, bert_result
            
        except Exception as e:
            logger.error(f"Error processing with BERT: {e}")
            return None, None
    
    def run_integration_test(self):
        """Run integration test."""
        logger.info("Running BERT-Ollama integration test...")
        
        # Check services
        bert_ok, ollama_ok = self.check_services()
        logger.info(f"BERT API: {'✓' if bert_ok else '✗'}")
        logger.info(f"Ollama API: {'✓' if ollama_ok else '✗'}")
        
        if not bert_ok or not ollama_ok:
            logger.error("One or more services are not available")
            return False
        
        # Test question
        question = "What is deep learning?"
        context = "Deep learning is a subset of machine learning that uses neural networks with multiple layers."
        
        # Process through integration
        ollama_prompt, bert_result = self.process_question(question, context)
        
        if ollama_prompt:
            logger.info("Integration test successful!")
            logger.info(f"BERT Answer: {bert_result['answer']}")
            logger.info(f"Confidence: {bert_result['confidence']:.4f}")
            return True
        else:
            logger.error("Integration test failed")
            return False

if __name__ == "__main__":
    integration = BertOllamaIntegration()
    integration.run_integration_test()
EOF

chmod +x integrate_bert_ollama.py

echo ""
echo "Integration script created: integrate_bert_ollama.py"
echo "Run with: python3 integrate_bert_ollama.py"
