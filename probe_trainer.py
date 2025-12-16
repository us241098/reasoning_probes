"""
Train simple linear classifiers (probes) to predict if a reasoning step is correct.

Just logistic regression. The idea is to see if the hidden states
contain enough information to tell if a step is right or wrong.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict


class ProbeTrainer:
    """Train a simple logistic regression classifier on the features."""
    
    def __init__(self, random_state: int = 42):
        """
        Set up the probe.
        
        Args:
            random_state: Random seed so results are reproducible
        """
        self.random_state = random_state
    
        self.probe = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='lbfgs',
            C=1.0  # Default L2 regularization
        )
        self.is_trained = False
    
    def train(self, hidden_states: np.ndarray, labels: np.ndarray, 
              test_size: float = 0.2) -> Dict:
        """
        Train the linear probe.
        
        Args:
            hidden_states: Array of hidden state vectors (n_samples, hidden_dim)
            labels: Binary labels (1 = correct, 0 = incorrect)
            test_size: Fraction of data to use for testing
        
        Returns:
            Dictionary with training results
        """
        # Check if we can do stratified split (need at least 2 samples per class)
        unique, counts = np.unique(labels, return_counts=True)
        min_class_count = counts.min()
        
        # Need at least 2 samples in minority class for stratified split
        use_stratify = min_class_count >= 2
        
        if not use_stratify:
            print(f"Warning: Class imbalance detected (min class has {min_class_count} samples). "
                  "Disabling stratified split.")
        
        # Split data (if test_size is None or 0, use all data for training)
        if test_size is None or test_size == 0.0:
            X_train, X_test, y_train, y_test = hidden_states, None, labels, None
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                hidden_states,
                labels,
                test_size=test_size,
                random_state=self.random_state,
                stratify=labels if use_stratify else None
            )
        
        # Train probe
        self.probe.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on training data
        train_pred = self.probe.predict(X_train)
        train_proba = self.probe.predict_proba(X_train)[:, 1]
        train_acc = accuracy_score(y_train, train_pred)
        
        # AUC requires both classes to be present
        if len(np.unique(y_train)) > 1:
            train_auc = roc_auc_score(y_train, train_proba)
        else:
            train_auc = float('nan')
            print("Warning: Only one class in train set, AUC undefined")
        
        # Evaluate on test set if we have one
        if X_test is not None and len(X_test) > 0:
            test_pred = self.probe.predict(X_test)
            test_proba = self.probe.predict_proba(X_test)[:, 1]
            test_acc = accuracy_score(y_test, test_pred)
            
            if len(np.unique(y_test)) > 1:
                test_auc = roc_auc_score(y_test, test_proba)
            else:
                test_auc = float('nan')
                print("Warning: Only one class in test set, AUC undefined")
        else:
            test_pred = None
            test_proba = None
            test_acc = None
            test_auc = None
        
        results = {
            'train_auc': train_auc,
            'train_accuracy': train_acc,
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'train_size': len(X_train),
            'test_size': len(X_test) if X_test is not None else 0,
        }
        
        # Only generate classification report if we have test data
        if y_test is not None and test_pred is not None:
            results['classification_report'] = classification_report(y_test, test_pred, output_dict=True)
        else:
            results['classification_report'] = None
        
        return results
    
    def evaluate(self, hidden_states: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Evaluate the trained probe on new data.
        
        Args:
            hidden_states: Array of hidden state vectors
            labels: Binary labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Probe must be trained before evaluation")
        
        pred = self.probe.predict(hidden_states)
        proba = self.probe.predict_proba(hidden_states)[:, 1]
        
        auc = roc_auc_score(labels, proba)
        acc = accuracy_score(labels, pred)
        
        return {
            'auc': auc,
            'accuracy': acc,
            'predictions': pred,
            'probabilities': proba
        }
    
    def plot_roc_curve(self, hidden_states: np.ndarray, labels: np.ndarray, 
                       save_path: str = None):
        """
        Plot ROC curve for the probe.
        
        Args:
            hidden_states: Array of hidden state vectors
            labels: Binary labels
            save_path: Path to save the plot
        """
        if not self.is_trained:
            raise ValueError("Probe must be trained before plotting")
        
        proba = self.probe.predict_proba(hidden_states)[:, 1]
        fpr, tpr, thresholds = roc_curve(labels, proba)
        auc = roc_auc_score(labels, proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Reasoning Error Detection Probe')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_feature_importance(self, top_n: int = 20, save_path: str = None):
        """
        Plot feature importance (coefficients) of the linear probe.
        
        Args:
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        if not self.is_trained:
            raise ValueError("Probe must be trained before plotting")
        
        coefficients = np.abs(self.probe.coef_[0])
        top_indices = np.argsort(coefficients)[-top_n:][::-1]
        top_coeffs = coefficients[top_indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_indices)), top_coeffs)
        plt.yticks(range(len(top_indices)), [f'Feature {i}' for i in top_indices])
        plt.xlabel('Absolute Coefficient Value')
        plt.title(f'Top {top_n} Most Important Features for Error Detection')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()

