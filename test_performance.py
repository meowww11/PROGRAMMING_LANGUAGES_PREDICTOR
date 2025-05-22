import os
import time
import psutil
import json
from pathlib import Path
from language_predictor import LanguagePredictor

def main():
    # Initialize predictor
    predictor = LanguagePredictor()
    
    # Load supported languages
    lang_path = os.path.join(os.path.dirname(__file__), 'languages.json')
    with open(lang_path, 'r') as f:
        languages = json.load(f)
    
    # Train the model
    print("Training model...")
    start_time = time.time()
    predictor.train('FileIdentWorkSample_1/FileTypeData')
    training_time = time.time() - start_time
    
    # Get memory usage
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Memory usage: {memory_usage:.2f} MB")
    
    # Validate the model
    print("\nValidating model...")
    val_results = predictor.validate('FileIdentWorkSample_1/FileTypeData')
    print(f"Validation accuracy: {val_results['accuracy']:.4f}")
    print("\nValidation report:")
    print(val_results['report'])
    
    # Test the model
    print("\nTesting model...")
    test_results = predictor.test('FileIdentWorkSample_1/FileTypeData')
    print(f"Test accuracy: {test_results['accuracy']:.4f}")
    print("\nTest report:")
    print(test_results['report'])
    
    # Print confidence statistics
    confidences = test_results['confidences']
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        print(f"\nConfidence statistics:")
        print(f"Average: {avg_confidence:.4f}")
        print(f"Minimum: {min_confidence:.4f}")
        print(f"Maximum: {max_confidence:.4f}")

if __name__ == "__main__":
    main() 