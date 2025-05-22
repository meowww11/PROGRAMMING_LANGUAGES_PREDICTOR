# Language Identification System

A system for identifying programming languages in source code files using machine learning. This implementation supports both a legacy TensorFlow-based model and a new scikit-learn-based model with improved performance and resource efficiency.

## Features

- Supports 8 programming languages: Python, Java, C++, Groovy, JavaScript, XML, JSON, and YAML  
- Backward compatibility with the original TensorFlow implementation  
- Resource-efficient implementation (runs on CPU with <512MB RAM)  
- Fast prediction (more than 4 files/second)  
- Handles class imbalance through balanced class weights  
- Provides confidence scores for predictions  
- Supports batch processing of multiple files  

## Requirements

- Python 3.6+  
- Dependencies listed in `requirements.txt`  

## Installation

1. Clone the repository  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt

## Usage

### Training the Model

```bash
python predict_lang.py --train --train-dir FileTypeData
```

### Predicting Language (CLI)

Single file:
```bash
python predict_lang.py --file path/to/file.txt
```

Directory (batch) prediction:
```bash
python predict_lang.py --dir path/to/directory/
```

Show top-k predictions:
```bash
python predict_lang.py --file path/to/file.txt --top-k 3
```                  ## Implementation Details

### Model Architecture

The new implementation uses:
- HashingVectorizer with character n-grams (1-3) for feature extraction
- SGDClassifier (logistic regression) with balanced class weights
- Smart resampling: majority classes capped at 500, minority classes oversampled to 200
- Lightweight, language-specific features (keywords, syntax markers, comment styles)
- Efficient memory usage and fast inference

### Resource Constraints

The implementation is optimized for:
- Single quad-core CPU
- 512MB RAM
- Local file storage
- Processing > 4 files/second

### Class Imbalance Handling

- Smart resampling for balanced training
- Class weights in the classifier
- Maintains prediction accuracy across all supported languages

## Results

- **Test accuracy:** ~93%
- **Validation accuracy:** ~94%
- **Macro F1-score:** ~0.78 (test set)
- **Weighted F1-score:** ~0.94 (test set)
- **Resource usage:** Peak memory ~139MB, training time ~75s, prediction speed >4 files/sec

## Limitations

1. May struggle with:
   - Very short files
   - Files with mixed languages
   - Files with non-standard extensions
   - Binary files or non-UTF-8 encoded files

2. Resource constraints:
   - Limited to CPU processing
   - Memory usage must stay under 512MB
   - Processing speed target of 4 files/second

## Future Improvements

1. Model improvements:
   - Add file extension as a feature
   - Implement ensemble methods
   - Add confidence thresholds
   - Create language-specific preprocessing rules

2. Performance optimizations:
   - Implement batch processing for training
   - Add caching for frequently accessed files
   - Optimize feature extraction pipeline

3. Additional features:
   - Support for more languages
   - Better handling of mixed-language files
   - Improved error handling and logging
   - API for integration with other tools

## Deliverables and Key Files

### Source Code (Core Scripts)
- `language_predictor.py` — Main model and feature extraction logic.
- `predict_lang.py` — Command-line interface for training and prediction.
- `split_data.py` — Script to split data into train/validation/test sets.
- `test_performance.py` — Script to evaluate model performance.
- `generate_and_test.py` — Script to generate and test sample files.

### Documentation
- `README.md` — Main documentation and usage instructions.

# Training the Model
python predict_lang.py --train --train-dir FileTypeData

# Predicting Language from a Single File
python predict_lang.py --file path/to/file.txt

# Predicting Languages for Multiple Files in a Directory
python predict_lang.py --dir path/to/directory/

# Showing Top-k Predictions
python predict_lang.py --file path/to/file.txt --top-k 3

