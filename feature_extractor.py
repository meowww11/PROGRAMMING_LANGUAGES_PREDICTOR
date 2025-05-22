from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib
import os

class FeatureExtractor:
    def __init__(self, ngram_range=(1, 3), max_features=10000):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            analyzer='char_wb',  # character n-grams with word boundaries
            lowercase=True
        )
        self.is_fitted = False

    def fit(self, texts):
        """Fit the vectorizer on training texts"""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts):
        """Transform texts to TF-IDF features"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transforming")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        """Fit and transform in one step"""
        return self.fit(texts).transform(texts)

    def save(self, path):
        """Save the vectorizer to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted vectorizer")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.vectorizer, path)

    @classmethod
    def load(cls, path):
        """Load a vectorizer from disk"""
        extractor = cls()
        extractor.vectorizer = joblib.load(path)
        extractor.is_fitted = True
        return extractor 