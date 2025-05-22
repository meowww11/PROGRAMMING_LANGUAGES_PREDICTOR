import argparse
import os
from pathlib import Path
from model_trainer import ModelTrainer
import time

def read_file(file_path):
    """Read file content with UTF-8 encoding"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        print(f"Warning: Could not read {file_path} as UTF-8. Skipping.")
        return None

def predict_file(model, file_path, top_k=3):
    """Predict language for a single file"""
    content = read_file(file_path)
    if content is None:
        return None
    
    start_time = time.time()
    probs = model.predict_proba([content])[0]
    prediction_time = time.time() - start_time
    
    # Get top-k predictions
    top_predictions = sorted(zip(model.languages.keys(), probs), 
                           key=lambda x: x[1], 
                           reverse=True)[:top_k]
    
    return {
        'predictions': top_predictions,
        'time': prediction_time
    }

def main():
    parser = argparse.ArgumentParser(description='Predict programming language of files')
    parser.add_argument('path', help='File or directory path to analyze')
    parser.add_argument('--model-dir', default='models', help='Directory containing trained model')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions to show')
    parser.add_argument('--batch', action='store_true', help='Process multiple files')
    args = parser.parse_args()

    # Load model
    try:
        model = ModelTrainer.load(args.model_dir)
    except FileNotFoundError:
        print(f"Error: Model not found in {args.model_dir}")
        return

    path = Path(args.path)
    if path.is_file():
        # Single file prediction
        result = predict_file(model, str(path), args.top_k)
        if result:
            print(f"\nFile: {path}")
            print(f"Prediction time: {result['time']:.3f}s")
            print("Top predictions:")
            for lang, prob in result['predictions']:
                print(f"  {lang}: {prob:.2%}")
    
    elif path.is_dir() and args.batch:
        # Batch processing
        total_files = 0
        total_time = 0
        
        for file_path in path.rglob('*'):
            if file_path.is_file():
                result = predict_file(model, str(file_path), args.top_k)
                if result:
                    total_files += 1
                    total_time += result['time']
                    
                    print(f"\nFile: {file_path}")
                    print("Top predictions:")
                    for lang, prob in result['predictions']:
                        print(f"  {lang}: {prob:.2%}")
        
        if total_files > 0:
            print(f"\nProcessed {total_files} files")
            print(f"Average prediction time: {total_time/total_files:.3f}s")
            print(f"Files per second: {total_files/total_time:.1f}")
    
    else:
        print("Error: Invalid path or missing --batch flag for directory")

if __name__ == '__main__':
    main() 