import os
from pathlib import Path
from language_predictor import LanguagePredictor

def generate_test_files():
    """Generate test files for each language."""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Test files with more complex, language-specific content
    test_files = {
        "test.py": """
import numpy as np
from typing import List, Dict, Optional

class DataProcessor:
    def __init__(self, data: List[float], config: Optional[Dict] = None):
        self.data = np.array(data)
        self.config = config or {}
    
    def process(self) -> Dict[str, float]:
        mean = np.mean(self.data)
        std = np.std(self.data)
        return {"mean": mean, "std": std}
    
    @staticmethod
    def validate_data(data: List[float]) -> bool:
        return all(isinstance(x, (int, float)) for x in data)

if __name__ == "__main__":
    processor = DataProcessor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = processor.process()
    print(f"Results: {result}")
""",
        "test.java": """
package com.example;

import java.util.List;
import java.util.Map;
import java.util.Optional;

public class DataProcessor {
    private final List<Double> data;
    private final Map<String, Object> config;
    
    public DataProcessor(List<Double> data, Map<String, Object> config) {
        this.data = data;
        this.config = Optional.ofNullable(config).orElse(Map.of());
    }
    
    public Map<String, Double> process() {
        double mean = data.stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0);
            
        double std = Math.sqrt(
            data.stream()
                .mapToDouble(x -> Math.pow(x - mean, 2))
                .average()
                .orElse(0.0)
        );
        
        return Map.of("mean", mean, "std", std);
    }
    
    public static boolean validateData(List<Double> data) {
        return data != null && !data.isEmpty();
    }
}
""",
        "test.cpp": """
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <optional>

class DataProcessor {
private:
    std::vector<double> data;
    std::map<std::string, std::any> config;
    
public:
    DataProcessor(const std::vector<double>& data, 
                 const std::optional<std::map<std::string, std::any>>& config = std::nullopt)
        : data(data), config(config.value_or(std::map<std::string, std::any>())) {}
    
    std::map<std::string, double> process() {
        double mean = 0.0;
        for (const auto& x : data) {
            mean += x;
        }
        mean /= data.size();
        
        double std = 0.0;
        for (const auto& x : data) {
            std += std::pow(x - mean, 2);
        }
        std = std::sqrt(std / data.size());
        
        return {{"mean", mean}, {"std", std}};
    }
    
    static bool validateData(const std::vector<double>& data) {
        return !data.empty();
    }
};
""",
        "test.groovy": """
package com.example

class DataProcessor {
    List<Double> data
    Map<String, Object> config
    
    DataProcessor(List<Double> data, Map<String, Object> config = [:]) {
        this.data = data
        this.config = config ?: [:]
    }
    
    Map<String, Double> process() {
        def mean = data.sum() / data.size()
        def std = Math.sqrt(data.collect { Math.pow(it - mean, 2) }.sum() / data.size())
        return [mean: mean, std: std]
    }
    
    static boolean validateData(List<Double> data) {
        data && data.every { it instanceof Double }
    }
}
""",
        "test.js": """
class DataProcessor {
    constructor(data, config = {}) {
        this.data = data;
        this.config = config;
    }
    
    process() {
        const mean = this.data.reduce((a, b) => a + b, 0) / this.data.length;
        const std = Math.sqrt(
            this.data.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / this.data.length
        );
        return { mean, std };
    }
    
    static validateData(data) {
        return Array.isArray(data) && data.every(x => typeof x === 'number');
    }
}

// Example usage
const processor = new DataProcessor([1, 2, 3, 4, 5]);
console.log(processor.process());
""",
        "test.xml": """
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <settings>
        <property name="debug" value="true"/>
        <property name="timeout" value="5000"/>
    </settings>
    <database>
        <connection>
            <url>jdbc:mysql://localhost:3306/mydb</url>
            <username>user</username>
            <password>secret</password>
        </connection>
        <pool>
            <min-size>5</min-size>
            <max-size>20</max-size>
        </pool>
    </database>
</configuration>
""",
        "test.json": """
{
    "configuration": {
        "settings": {
            "debug": true,
            "timeout": 5000
        },
        "database": {
            "connection": {
                "url": "jdbc:mysql://localhost:3306/mydb",
                "username": "user",
                "password": "secret"
            },
            "pool": {
                "min-size": 5,
                "max-size": 20
            }
        }
    }
}
""",
        "test.yaml": """
configuration:
  settings:
    debug: true
    timeout: 5000
  database:
    connection:
      url: jdbc:mysql://localhost:3306/mydb
      username: user
      password: secret
    pool:
      min-size: 5
      max-size: 20
"""
    }
    
    print("Generating test files...")
    for filename, content in test_files.items():
        filepath = test_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"Generated {filepath}")

def test_predictor():
    """Test the language predictor on generated test files."""
    # Load the pre-trained model
    print("\nLoading pre-trained model...")
    predictor = LanguagePredictor.load(model_dir='models')
    
    # Test each file
    print("\nTesting files...")
    test_dir = Path("test_data")
    results = []
    
    for filepath in test_dir.glob("test.*"):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        true_lang = filepath.suffix[1:]  # Remove the dot
        
        # Get prediction with confidence
        predictions = predictor.predict(content, top_k=3)  # Get top 3 predictions
        pred_lang = predictions[0][0]  # Top prediction
        confidence = predictions[0][1]  # Confidence of top prediction
        
        results.append({
            "file": filepath.name,
            "true_lang": true_lang,
            "pred_lang": pred_lang,
            "confidence": confidence,
            "correct": true_lang.lower() == pred_lang.lower(),
            "content": content,
            "alt_predictions": predictions[1:]  # Store alternative predictions
        })
    
    # Print results
    print("\nTest Results:")
    print("-" * 80)
    print(f"{'File':<15} {'True Lang':<10} {'Predicted':<10} {'Confidence':<10} {'Correct':<8} {'Alt Predictions'}")
    print("-" * 80)
    
    for result in results:
        alt_str = ", ".join(f"{lang}({conf:.2%})" for lang, conf in result["alt_predictions"])
        
        print(f"{result['file']:<15} {result['true_lang']:<10} {result['pred_lang']:<10} "
              f"{result['confidence']:.2%} {str(result['correct']):<8} {alt_str}")
    
    # Calculate accuracy
    accuracy = sum(1 for r in results if r['correct']) / len(results)
    print("\nOverall Accuracy:", f"{accuracy:.2%}")

if __name__ == "__main__":
    generate_test_files()
    test_predictor() 