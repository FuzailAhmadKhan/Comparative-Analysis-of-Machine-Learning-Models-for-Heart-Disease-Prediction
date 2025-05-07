import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score, auc
)
import warnings
import time
from sklearn.inspection import permutation_importance  # For universal feature importance

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.precision', 4)

class ModelEvaluator:
    def __init__(self, data_path: str = 'preprocessed_data.npz'):
        self.data = np.load(data_path)
        self.X_test = self.data['X_test']
        self.y_test = self.data['y_test']
        self.feature_names = self.data.get('feature_names', 
                                         [f'Feature {i}' for i in range(self.X_test.shape[1])])
        self.results = pd.DataFrame(columns=[
            'Model', 'Accuracy', 'Precision', 'Recall', 'F1',
            'ROC AUC', 'Avg Precision', 'Inference Time'
        ])
        self.models = {}

    def load_models(self, model_paths: dict):
        """Load models with perfectly cleaned names"""
        self.models = {
            ' '.join(word.capitalize() for word in name.replace('1m', '').split('_')): joblib.load(path)
            for name, path in model_paths.items()
        }

    def evaluate(self):
        """Run comprehensive evaluation with guaranteed visualizations"""
        print("\n" + "="*40)
        print("  MODEL EVALUATION REPORT  ".center(40, '='))
        print("="*40 + "\n")
        
        for name, model in self.models.items():
            print(f"\n\033[1m{name.upper()}\033[0m")
            print("=" * len(name)*2)
            
            # Make predictions
            start_time = time.time()
            y_pred = model.predict(self.X_test)
            inference_time = time.time() - start_time
            
            # Get probabilities (works for all models)
            y_prob = self._get_probabilities(model, y_pred)
            
            # Calculate metrics
            metrics = self._calculate_metrics(name, y_pred, y_prob, inference_time)
            self.results = pd.concat([self.results, pd.DataFrame([metrics])], ignore_index=True)
            
            # Create visualizations (now with guaranteed feature importance)
            self._create_model_dashboard(name, model, y_pred, y_prob)
            
            # Print metrics
            self._print_metrics(metrics, y_pred)

        self._display_final_results()

    def _get_probabilities(self, model, y_pred):
        """Universal probability/score extraction"""
        try:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(self.X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                decision = model.decision_function(self.X_test)
                return (decision - decision.min()) / (decision.max() - decision.min())
            else:  # For models without probability (use normalized predictions)
                return (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
        except:
            return None

    def _calculate_metrics(self, name, y_pred, y_prob, inference_time):
        """Calculate all metrics with validation"""
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1': f1_score(self.y_test, y_pred),
            'Inference Time': f"{inference_time:.4f}s"
        }
        
        if y_prob is not None:
            try:
                metrics['ROC AUC'] = roc_auc_score(self.y_test, y_prob)
                metrics['Avg Precision'] = average_precision_score(self.y_test, y_prob)
            except:
                metrics['ROC AUC'] = np.nan
                metrics['Avg Precision'] = np.nan
        else:
            metrics['ROC AUC'] = "N/A"
            metrics['Avg Precision'] = "N/A"
            
        return metrics

    def _get_feature_importance(self, model):
        """Universal feature importance calculation"""
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            else:  # Use permutation importance as fallback
                r = permutation_importance(model, self.X_test, self.y_test, n_repeats=10)
                return r.importances_mean
        except:
            return np.zeros(self.X_test.shape[1])  # Return zeros if all fails

    def _create_model_dashboard(self, name, model, y_pred, y_prob):
        """Create perfect visualization dashboard"""
        plt.close('all')
        fig = plt.figure(figsize=(16, 12), facecolor='#f5f5f5')
        
        # Main title with beautiful formatting - ONLY CHANGED THIS SECTION
        fig.suptitle(f"{name} Evaluation Dashboard", 
                    fontsize=18, fontweight='bold', y=0.98,
                    color='#2e3440')
        
        # Create grid layout
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
        
        # Plot 1: Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Healthy', 'Heart Disease'],
                   yticklabels=['Healthy', 'Heart Disease'],
                   cbar_kws={'label': 'Count'})
        ax1.set_title('Confusion Matrix', pad=12, fontweight='bold')
        ax1.set_xlabel('Predicted', fontsize=10)
        ax1.set_ylabel('Actual', fontsize=10)
        
        # Plot 2: ROC Curve
        ax2 = fig.add_subplot(gs[0, 1])
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, color='#5e81ac', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax2.plot([0, 1], [0, 1], color='#d08770', linestyle='--')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.legend(loc="lower right", framealpha=0.9)
        else:
            ax2.text(0.5, 0.5, 'ROC Curve\nNot Available', 
                    ha='center', va='center', fontsize=12)
            ax2.set_xticks([])
            ax2.set_yticks([])
        ax2.set_title('ROC Curve', pad=12, fontweight='bold')
        ax2.set_xlabel('False Positive Rate', fontsize=10)
        ax2.set_ylabel('True Positive Rate', fontsize=10)
        
        # Plot 3: Precision-Recall Curve
        ax3 = fig.add_subplot(gs[1, 0])
        if y_prob is not None:
            precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
            avg_precision = average_precision_score(self.y_test, y_prob)
            ax3.plot(recall, precision, color='#88c0d0', lw=2,
                    label=f'Avg Precision = {avg_precision:.2f}')
            ax3.legend(loc="upper right", framealpha=0.9)
        else:
            ax3.text(0.5, 0.5, 'Precision-Recall\nNot Available', 
                    ha='center', va='center', fontsize=12)
            ax3.set_xticks([])
            ax3.set_yticks([])
        ax3.set_title('Precision-Recall Curve', pad=12, fontweight='bold')
        ax3.set_xlabel('Recall', fontsize=10)
        ax3.set_ylabel('Precision', fontsize=10)
        
        # Plot 4: Feature Importance (now guaranteed for all models)
        ax4 = fig.add_subplot(gs[1, 1])
        importances = self._get_feature_importance(model)
        indices = np.argsort(importances)[::-1]
        
        bars = ax4.bar(range(len(importances)), importances[indices],
                      color='#a3be8c', align="center")
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8)
        
        ax4.set_xticks(range(len(importances)))
        ax4.set_xticklabels([self.feature_names[i] for i in indices], 
                           rotation=90, fontsize=8)
        ax4.set_xlim([-1, len(importances)])
        ax4.set_title('Feature Importance', pad=12, fontweight='bold')
        ax4.set_ylabel('Importance Score', fontsize=10)
        
        plt.tight_layout()
        plt.show()

    def _print_metrics(self, metrics: dict, y_pred: np.ndarray):
        """Beautifully formatted metric printing"""
        print("\n\033[1mPERFORMANCE METRICS\033[0m")
        print("-" * 40)
        print(f"{'Accuracy:':<20}{metrics['Accuracy']:.4f}")
        print(f"{'Precision:':<20}{metrics['Precision']:.4f}")
        print(f"{'Recall:':<20}{metrics['Recall']:.4f}")
        print(f"{'F1 Score:':<20}{metrics['F1']:.4f}")
        print(f"{'ROC AUC:':<20}{metrics['ROC AUC']}")
        print(f"{'Avg Precision:':<20}{metrics['Avg Precision']}")
        print(f"{'Inference Time:':<20}{metrics['Inference Time']}")
        
        print("\n\033[1mCLASSIFICATION REPORT\033[0m")
        print("-" * 40)
        print(classification_report(self.y_test, y_pred, digits=4))

    def _display_final_results(self):
        """Final results with perfect formatting"""
        print("\n" + "="*60)
        print("  FINAL MODEL COMPARISON  ".center(60, '='))
        print("="*60)
        
        final_results = self.results.sort_values(by='Accuracy', ascending=False)
        print(final_results.to_string(index=False, justify='center', float_format='%.4f'))
        
        final_results.to_csv('model_evaluation_results.csv', index=False)
        print("\n\033[1mResults saved to 'model_evaluation_results.csv'\033[0m")

if __name__ == "__main__":
    MODEL_PATHS = {
        'logistic_regression': 'logistic_regression_model.pkl',
        'decision_tree': 'decision_tree_model.pkl',
        'random_forest': 'random_forest_model.pkl',
        'svm': 'svm_model.pkl',
        'knn': 'knn_model.pkl'
    }
    
    evaluator = ModelEvaluator()
    evaluator.load_models(MODEL_PATHS)
    evaluator.evaluate()