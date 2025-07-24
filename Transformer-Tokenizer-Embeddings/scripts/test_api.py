#!/usr/bin/env python3
"""
Test script for BERT QA API using cURL requests.
This script demonstrates how to interact with the BERT QA API.
"""

import requests
import json
import time
import sys

class BertQAAPITester:
    """Test client for BERT QA API."""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def test_health_check(self):
        """Test the health check endpoint."""
        print("Testing health check...")
        try:
            response = requests.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                print("✓ Health check passed")
                print(f"Response: {response.json()}")
                return True
            else:
                print(f"✗ Health check failed: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("✗ Cannot connect to API server. Is it running?")
            return False
        except Exception as e:
            print(f"✗ Health check error: {e}")
            return False
    
    def test_model_info(self):
        """Test the model info endpoint."""
        print("\nTesting model info...")
        try:
            response = requests.get(f"{self.base_url}/api/model/info")
            if response.status_code == 200:
                print("✓ Model info retrieved")
                print(f"Response: {json.dumps(response.json(), indent=2)}")
                return True
            else:
                print(f"✗ Model info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Model info error: {e}")
            return False
    
    def test_question_answering(self):
        """Test the question answering endpoint."""
        print("\nTesting question answering...")
        
        test_cases = [
            {
                "question": "What is machine learning?",
                "context": "Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves."
            },
            {
                "question": "What is deep learning?",
                "context": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers (hence 'deep') to model and understand complex patterns in data. These neural networks are inspired by the structure and function of the human brain."
            },
            {
                "question": "What is BERT?",
                "context": "BERT (Bidirectional Encoder Representations from Transformers) is a revolutionary natural language processing model developed by Google. Unlike traditional language models that read text sequentially, BERT reads text bidirectionally, considering both left and right context simultaneously."
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Question: {test_case['question']}")
            
            try:
                response = requests.post(
                    f"{self.base_url}/api/qa",
                    json=test_case,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print("✓ Question answered successfully")
                    print(f"Answer: {result['answer']}")
                    print(f"Confidence: {result['confidence']:.4f}")
                    print(f"Metadata: {result['metadata']}")
                else:
                    print(f"✗ Question answering failed: {response.status_code}")
                    print(f"Error: {response.text}")
                    return False
                    
            except Exception as e:
                print(f"✗ Question answering error: {e}")
                return False
        
        return True
    
    def test_batch_qa(self):
        """Test batch question answering."""
        print("\nTesting batch question answering...")
        
        batch_data = {
            "qa_pairs": [
                {
                    "question": "What is AI?",
                    "context": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans."
                },
                {
                    "question": "What is NLP?",
                    "context": "Natural Language Processing (NLP) is a branch of artificial intelligence that deals with the interaction between computers and humans using natural language."
                }
            ]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/qa/batch",
                json=batch_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✓ Batch QA successful")
                print(f"Processed {result['batch_size']} questions")
                for i, res in enumerate(result['results'], 1):
                    print(f"  {i}. Q: {res['question']}")
                    print(f"     A: {res['answer']}")
                    print(f"     Confidence: {res['confidence']:.4f}")
                return True
            else:
                print(f"✗ Batch QA failed: {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Batch QA error: {e}")
            return False
    
    def test_error_handling(self):
        """Test error handling."""
        print("\nTesting error handling...")
        
        # Test missing fields
        try:
            response = requests.post(
                f"{self.base_url}/api/qa",
                json={"question": "What is AI?"},  # Missing context
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 400:
                print("✓ Missing field error handled correctly")
            else:
                print(f"✗ Expected 400 error, got {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Error handling test failed: {e}")
            return False
        
        # Test empty content
        try:
            response = requests.post(
                f"{self.base_url}/api/qa",
                json={"question": "", "context": ""},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 400:
                print("✓ Empty content error handled correctly")
            else:
                print(f"✗ Expected 400 error, got {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Error handling test failed: {e}")
            return False
        
        return True
    
    def run_all_tests(self):
        """Run all tests."""
        print("=" * 50)
        print("BERT QA API Test Suite")
        print("=" * 50)
        
        tests = [
            self.test_health_check,
            self.test_model_info,
            self.test_question_answering,
            self.test_batch_qa,
            self.test_error_handling
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
            time.sleep(1)  # Brief pause between tests
        
        print("\n" + "=" * 50)
        print(f"Test Results: {passed}/{total} tests passed")
        print("=" * 50)
        
        return passed == total

def print_curl_examples():
    """Print cURL examples for manual testing."""
    print("\n" + "=" * 50)
    print("cURL Examples for Manual Testing")
    print("=" * 50)
    
    examples = [
        {
            "name": "Health Check",
            "command": "curl -X GET http://localhost:5000/api/health"
        },
        {
            "name": "Model Info",
            "command": "curl -X GET http://localhost:5000/api/model/info"
        },
        {
            "name": "Question Answering",
            "command": """curl -X POST http://localhost:5000/api/qa \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "What is machine learning?",
    "context": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed."
  }'"""
        },
        {
            "name": "Batch QA",
            "command": """curl -X POST http://localhost:5000/api/qa/batch \\
  -H "Content-Type: application/json" \\
  -d '{
    "qa_pairs": [
      {
        "question": "What is AI?",
        "context": "Artificial Intelligence is the simulation of human intelligence in machines."
      },
      {
        "question": "What is ML?",
        "context": "Machine Learning is a subset of AI that enables systems to learn from data."
      }
    ]
  }'"""
        }
    ]
    
    for example in examples:
        print(f"\n{example['name']}:")
        print(example['command'])

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test BERT QA API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:5000",
        help="Base URL of the API server"
    )
    parser.add_argument(
        "--curl-examples",
        action="store_true",
        help="Print cURL examples instead of running tests"
    )
    
    args = parser.parse_args()
    
    if args.curl_examples:
        print_curl_examples()
    else:
        tester = BertQAAPITester(args.url)
        success = tester.run_all_tests()
        
        if not success:
            print("\nSome tests failed. Check the API server and try again.")
            sys.exit(1)
        else:
            print("\n✓ All tests passed successfully!")
            print("\nAPI is ready for use!")
        
        # Also print cURL examples
        print_curl_examples()
