import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
import time
import os
import pandas as pd
from datetime import datetime
from collections.abc import Iterable
from typing import List

class TrainingPlotter:
    def __init__(self):
        """
        Integrated training progress visualization tool (designed for Jupyter Notebook)
        
        Features:
        - Combines training loss, training accuracy, and test accuracy in one chart
        - Dual Y-axis design: loss on left, accuracy on right
        - Preserves final chart after training completes
        - Supports real-time updates and final report
        """
        # Initialize data records
        self.reset_history()
        
        # Create figure with dual Y-axes
        self.fig, self.ax1 = plt.subplots(figsize=(12, 6))
        self.ax2 = self.ax1.twinx()  # Share X-axis, create second Y-axis
        
        # Configure axes
        self._setup_axes()
        
        # Initialize plot lines
        self.line_loss, = self.ax1.plot([], [], 'b-', marker='o', label='Training Loss', linewidth=1.5)
        self.line_train_acc, = self.ax2.plot([], [], 'g-', marker='s', label='Training Accuracy', linewidth=1.5)
        self.line_test_acc, = self.ax2.plot([], [], 'r-', marker='^', label='Test Accuracy', linewidth=1.5)
        
        # Add legend
        self._add_legend()
        
        # Display initial chart
        self.fig.tight_layout()
        display(self.fig)
        
    def _add_legend(self):
        """Add legend to the chart"""
        # Get all lines and labels
        lines = [self.line_loss, self.line_train_acc, self.line_test_acc]
        # 显式创建字符串标签列表
        labels: List[str] = [
            str(self.line_loss.get_label()),
            str(self.line_train_acc.get_label()),
            str(self.line_test_acc.get_label())
        ]
        
        # Create legend
        self.ax1.legend(lines, labels, loc='upper left')
    
    def reset_history(self):
        """Reset all historical records"""
        self.epochs = []
        self.train_losses = []
        self.train_accs = []
        self.test_accs = []
    
    def _setup_axes(self):
        """Configure axes properties"""
        # Left Y-axis (Loss)
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss', color='b')
        self.ax1.tick_params(axis='y', labelcolor='b')
        
        # Right Y-axis (Accuracy)
        self.ax2.set_ylabel('Accuracy (%)', color='g')
        self.ax2.tick_params(axis='y', labelcolor='g')
        self.ax2.set_ylim(0, 100)  # Accuracy range 0-100%
        
        # Chart title
        self.ax1.set_title('Training Progress Monitoring')
        
        # Grid lines
        self.ax1.grid(True, linestyle='--', alpha=0.7)
    
    def update(self, epoch, train_loss, train_acc, test_acc):
        """
        Update training records and refresh chart
        
        Parameters:
            epoch (int): Current epoch
            train_loss (float): Training loss
            train_acc (float): Training accuracy (0-100)
            test_acc (float): Test accuracy (0-100)
        """
        # Record data
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.test_accs.append(test_acc)
        
        # Update chart
        self._update_plot()
    
    def _update_plot(self):
        """Update chart data and refresh display"""
        # Update line data
        self.line_loss.set_data(self.epochs, self.train_losses)
        self.line_train_acc.set_data(self.epochs, self.train_accs)
        self.line_test_acc.set_data(self.epochs, self.test_accs)
        
        # Adjust axis ranges
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.set_ylim(0, 100)  # Ensure accuracy stays in 0-100% range
        
        # Add current value labels
        self._add_value_labels()
        
        # Refresh display
        clear_output(wait=True)  # Clear previous chart but keep final chart
        display(self.fig)
    
    def _add_value_labels(self):
        """Add labels for the latest values on the chart"""
        # Remove old labels
        for artist in self.ax1.texts + self.ax2.texts:
            artist.remove()
        
        if self.epochs:
            last_epoch = self.epochs[-1]
            
            # Loss label (left)
            last_loss = self.train_losses[-1]
            self.ax1.annotate(f'Loss: {last_loss:.4f}', 
                             xy=(last_epoch, last_loss),
                             xytext=(5, 5), textcoords='offset points',
                             color='b', fontsize=9, weight='bold')
            
            # Training accuracy label (right)
            last_train_acc = self.train_accs[-1]
            self.ax2.annotate(f'Train: {last_train_acc:.2f}%', 
                             xy=(last_epoch, last_train_acc),
                             xytext=(5, 5), textcoords='offset points',
                             color='g', fontsize=9, weight='bold')
            
            # Test accuracy label (right)
            last_test_acc = self.test_accs[-1]
            self.ax2.annotate(f'Test: {last_test_acc:.2f}%', 
                             xy=(last_epoch, last_test_acc),
                             xytext=(5, -15), textcoords='offset points',
                             color='r', fontsize=9, weight='bold')
    
    def save_plot(self, filename=None):
        """Save chart to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_plot_{timestamp}.png"
        
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {filename}")
        return filename
    
    def final_report(self):
        """Generate final report and mark best results"""
        if not self.epochs:
            print("No training data available")
            return
        
        # Find best test accuracy
        best_test_idx = np.argmax(self.test_accs)
        best_test_acc = self.test_accs[best_test_idx]
        best_test_epoch = self.epochs[best_test_idx]
        
        # Find lowest training loss
        best_loss_idx = np.argmin(self.train_losses)
        best_loss = self.train_losses[best_loss_idx]
        best_loss_epoch = self.epochs[best_loss_idx]
        
        # Mark best results on chart
        self.ax2.plot(best_test_epoch, best_test_acc, 'r*', markersize=12)
        self.ax1.plot(best_loss_epoch, best_loss, 'b*', markersize=12)
        
        # Add annotations
        self.ax2.annotate(f'Best Test: {best_test_acc:.2f}% @ Epoch {best_test_epoch+1}',
                         xy=(best_test_epoch, best_test_acc),
                         xytext=(5, -25), textcoords='offset points',
                         color='r', fontsize=10, weight='bold')
        
        self.ax1.annotate(f'Lowest Loss: {best_loss:.4f} @ Epoch {best_loss_epoch+1}',
                         xy=(best_loss_epoch, best_loss),
                         xytext=(5, 5), textcoords='offset points',
                         color='b', fontsize=10, weight='bold')
        
        # Refresh final chart
        self._update_plot()
        
        # Generate report data
        summary_data = {
            'Total Epochs': len(self.epochs),
            'Final Training Loss': f"{self.train_losses[-1]:.4f}",
            'Lowest Training Loss': f"{best_loss:.4f} (epoch {best_loss_epoch+1})",
            'Final Training Accuracy': f"{self.train_accs[-1]:.2f}%",
            'Final Test Accuracy': f"{self.test_accs[-1]:.2f}%",
            'Best Test Accuracy': f"{best_test_acc:.2f}% (epoch {best_test_epoch+1})",
            'Test Accuracy Improvement': f"{(self.test_accs[-1] - self.test_accs[0]):.2f}%",
            'Overfitting Gap': f"{(self.train_accs[-1] - self.test_accs[-1]):.2f}%"
        }
        
        # Create formatted report table
        print("\n" + "="*70)
        print("TRAINING SUMMARY REPORT".center(70))
        print("="*70)
        
        # Convert to DataFrame for pretty printing
        df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Value'])
        print(df.to_string(index=False))
        print("="*70)
        
        return summary_data