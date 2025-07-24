# BERT QA

## Setup & run

### 1. Setup environment (5 minutes)
```bash
# Run the setup script
./scripts/setup_environment.sh

# Activate the virtual environment
source bert_qa_env/bin/activate
```

### 2. Test the API (Without Fine-tuning)
```bash
# Start the API server (uses mock model initially)
python app.py &

# Test the API
python scripts/test_api.py

# Or test with cURL
curl -X POST http://localhost:5000/api/qa \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "context": "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
  }'
```

### 3. Fine-tune BERT model (Optional - 10-15 minutes)
```bash
# Run fine-tuning with limited resources
python scripts/train_model.py \
  --subset_size 500 \
  --num_epochs 1 \
  --max_steps 200 \
  --batch_size 2
```

### 4. Deploy with Ollama (Optional)
```bash
# Setup Ollama with Docker
./scripts/deploy_ollama.sh

# Test integration
python integrate_bert_ollama.py
```

## 📋 Step-by-step

### Prerequisites
- Python 3.8+
- pip
- Virtual environment support
- Docker (for Ollama deployment)
- 4GB+ RAM recommended

### Environment setup
1. **Clone/Navigate to project directory**
2. **Run setup script**: `./scripts/setup_environment.sh`
3. **Activate environment**: `source bert_qa_env/bin/activate`

### Usage
1. **Start API**: `python app.py`
2. **Test endpoints**: `python scripts/test_api.py`
3. **Try cURL examples** (see output from test script)

### Fine-tuning (For better results)
1. **Run training**: `python scripts/train_model.py`
2. **Monitor progress**: Check logs in `training.log`
3. **Test fine-tuned model**: Restart API and test again

### Docker deployment
```bash
# Build and run with Docker Compose
cd docker
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs bert-qa-api
```

## Configuration

### Training parameters
- `--subset_size`: Number of training examples (default: 1000)
- `--num_epochs`: Training epochs (default: 2)
- `--max_steps`: Maximum training steps (default: 500)
- `--batch_size`: Batch size (default: 4)
- `--learning_rate`: Learning rate (default: 2e-5)

### API configuration
- Port: 5000 (change in `app.py`)
- Debug mode: Set `DEBUG=true` environment variable
- Model path: Configurable in API initialization

## Testing

### See PEFT in action
```bash
# Demonstrate PEFT benefits vs traditional fine-tuning
python scripts/demonstrate_peft.py
```

### Automated tests
```bash
python scripts/test_api.py
```

### Manual cURL tests
```bash
# Health check
curl http://localhost:5000/api/health

# Single question
curl -X POST http://localhost:5000/api/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AI?", "context": "AI is..."}'

# Batch questions
curl -X POST http://localhost:5000/api/qa/batch \
  -H "Content-Type: application/json" \
  -d '{"qa_pairs": [{"question": "What is ML?", "context": "ML is..."}]}'
```

## Performance

### For limited resources:
1. Use smaller subset sizes (`--subset_size 200`)
2. Reduce batch size (`--batch_size 2`)
3. Limit training steps (`--max_steps 100`)
4. Use CPU-only PyTorch
5. Enable quantization (default: enabled)

### For better results:
1. Use larger datasets
2. Increase training epochs
3. Fine-tune hyperparameters
4. Use GPU if available
5. Experiment with different BERT variants

## Troubleshooting

### Issues:

**Import errors**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

**CUDA errors**
- Use CPU version: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

**Memory issues**
- Reduce batch size
- Use smaller subset
- Enable gradient checkpointing

**API connection issues**
- Check if server is running: `curl http://localhost:5000/api/health`
- Verify port 5000 is not in use

**Docker issues**
- Ensure Docker is running
- Check container logs: `docker logs bert-qa-api`

## 📁 Project Structure
```
├── app.py                  # Flask REST API
├── bert_qa.py              # BERT QA implementation
├── fine_tune.py            # Fine-tuning script
├── scripts/
│   ├── setup_environment.sh # Environment setup
│   ├── train_model.py      # Training wrapper
│   ├── test_api.py         # API testing
│   └── deploy_ollama.sh    # Ollama deployment
├── data/
│   └── sample_qa_data.json # Sample QA data
├── docker/
│   ├── Dockerfile          # Container definition
│   └── docker-compose.yml  # Multi-service setup
└── requirements.txt        # Python dependencies
```

## Next steps

1. **Experiment with different models**: Try `bert-base-uncased`, `roberta-base`
2. **Custom datasets**: Replace SQuAD with domain-specific data
3. **Production deployment**: Use Gunicorn, NGINX, load balancing
4. **Model optimization**: Explore ONNX, TensorRT for faster inference
5. **Monitoring**: Add logging, metrics, health checks

## Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Ollama Documentation](https://ollama.ai/docs)

---
