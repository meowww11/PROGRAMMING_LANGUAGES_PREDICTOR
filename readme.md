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
