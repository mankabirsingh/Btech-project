"""
Baseline Model: TF-IDF + Logistic Regression
=============================================
Lightweight vulnerability detection baseline.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


class VulnerabilityBaseline:
    """
    TF-IDF + Logistic Regression baseline for vulnerability detection.
    
    Uses character and word n-grams from source code to predict
    whether a code function is vulnerable or benign.
    """
    
    def __init__(
        self,
        max_features: int = 30000,
        ngram_range: Tuple[int, int] = (1, 3),
        analyzer: str = 'char_wb',
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = 'balanced',
        random_seed: int = 42,
    ):
        """
        Args:
            max_features: Maximum vocabulary size for TF-IDF
            ngram_range: N-gram range for feature extraction
            analyzer: 'char_wb', 'char', or 'word'
            C: Regularization strength (smaller = more regularization)
            max_iter: Max iterations for logistic regression
            class_weight: 'balanced' to handle class imbalance
            random_seed: Random seed
        """
        self.config = {
            'max_features': max_features,
            'ngram_range': ngram_range,
            'analyzer': analyzer,
            'C': C,
            'max_iter': max_iter,
            'class_weight': class_weight,
            'random_seed': random_seed,
        }
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            analyzer=analyzer,
            sublinear_tf=True,
            strip_accents='unicode',
            decode_error='replace',
        )
        
        self.classifier = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_seed,
            solver='lbfgs',
            n_jobs=-1,
        )
        
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('clf', self.classifier),
        ])
        
        self.is_trained = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train the baseline model.
        
        Args:
            X_train: Array of code strings
            y_train: Binary labels (0/1)
            X_val: Validation code strings (optional)
            y_val: Validation labels (optional)
        
        Returns:
            Training info dict
        """
        logger.info(f"Training baseline: {len(X_train)} samples, "
                     f"vuln={sum(y_train)}, benign={len(y_train)-sum(y_train)}")
        
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        info = {
            'n_train': len(X_train),
            'n_vuln_train': int(sum(y_train)),
            'n_benign_train': int(len(y_train) - sum(y_train)),
            'vocab_size': len(self.vectorizer.vocabulary_),
        }
        
        # Training accuracy
        train_pred = self.pipeline.predict(X_train)
        info['train_accuracy'] = float(np.mean(train_pred == y_train))
        
        if X_val is not None and y_val is not None:
            val_pred = self.pipeline.predict(X_val)
            info['val_accuracy'] = float(np.mean(val_pred == y_val))
            logger.info(f"Train acc: {info['train_accuracy']:.4f}, "
                         f"Val acc: {info['val_accuracy']:.4f}")
        
        return info
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict vulnerability labels."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict vulnerability probabilities."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        return self.pipeline.predict_proba(X)
    
    def save(self, path: Path):
        """Save trained model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'config': self.config,
                'is_trained': self.is_trained,
            }, f)
        
        logger.info(f"Baseline model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'VulnerabilityBaseline':
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(**data['config'])
        model.pipeline = data['pipeline']
        model.vectorizer = model.pipeline.named_steps['tfidf']
        model.classifier = model.pipeline.named_steps['clf']
        model.is_trained = data['is_trained']
        
        logger.info(f"Baseline model loaded from {path}")
        return model


class WordLevelBaseline(VulnerabilityBaseline):
    """Word-level TF-IDF variant for comparison."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('analyzer', 'word')
        kwargs.setdefault('ngram_range', (1, 2))
        kwargs.setdefault('max_features', 20000)
        super().__init__(**kwargs)
