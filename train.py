import os
from pathlib import Path
import json
from model_trainer import ModelTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

def load_training_data(data_dir):
    """Load training data from the provided directory"""
    texts = []
    labels = []
    
    with open('languages.json') as f:
        languages = json.load(f)
    
    for lang, extensions in languages.items():
        for ext in extensions:
            # Find all files with this extension
            for file_path in Path(data_dir).rglob(f'*{ext}'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():  # Skip empty files
                            texts.append(content)
                            labels.append(lang)
                except UnicodeDecodeError:
                    print(f"Warning: Could not read {file_path} as UTF-8. Skipping.")
    
    return texts, labels

def main():
    # Load training data
    print("Loading training data...")
    texts, labels = load_training_data('FileTypeData')
    
    if not texts:
        print("Error: No training data found")
        return
    
    print(f"Loaded {len(texts)} training samples")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Train model
    print("\nTraining model...")
    start_time = time.time()
    model = ModelTrainer()
    model.train(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate on test set
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    print("\nSaving model...")
    model.save()
    
    print(f"\nTraining completed in {training_time:.1f} seconds")
    print(f"Model saved to {model.model_dir}")

if __name__ == '__main__':
    main() 