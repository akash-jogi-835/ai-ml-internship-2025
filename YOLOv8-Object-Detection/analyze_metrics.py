import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
from pathlib import Path

class YOLOAnalyzer:
    """Enhanced YOLO training results analyzer with comprehensive reporting"""
    
    def __init__(self, style='seaborn'):
        """Initialize analyzer with plotting style"""
        self.set_plot_style(style)
        
    def set_plot_style(self, style='seaborn'):
        """Set matplotlib plotting style"""
        plt.style.use(style)
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    def analyze_results(self, path, mode="detection", save_plots=False, output_dir=None):
        """
        Comprehensive analysis of YOLO training results
        
        Args:
            path (str): Path to results CSV file
            mode (str): Type of analysis - "detection" or "segmentation"
            save_plots (bool): Whether to save plots to file
            output_dir (str): Directory to save plots (defaults to results directory)
        """
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE {mode.upper()} TRAINING ANALYSIS")
        print(f"{'='*80}")
        
        if not os.path.exists(path):
            print(f"No results file found at: {path}")
            return None
        
        try:
            df = pd.read_csv(path)
            print(f"Loaded {len(df)} entries from {path}")
            
            # Set output directory for saving plots
            if save_plots and output_dir is None:
                output_dir = os.path.dirname(path)
            
            # Analyze based on file structure
            if 'metrics/mAP50(B)' in df.columns:
                self._analyze_yolo_metrics(df, mode, save_plots, output_dir)
            elif 'total_images' in df.columns:
                self._analyze_custom_metrics(df, mode, save_plots, output_dir)
            else:
                print("Unrecognized CSV structure – attempting basic analysis...")
                self._basic_analysis(df, mode, save_plots, output_dir)
                
            return df
            
        except Exception as e:
            print(f"Error analyzing {mode} metrics: {e}")
            return None
    
    def _analyze_yolo_metrics(self, df, mode, save_plots, output_dir):
        """Analyze standard YOLO training metrics"""
        print(f"\nYOLO {mode.upper()} METRICS ANALYSIS")
        print("-" * 50)

        # Basic statistics
        epochs_completed = len(df)
        print(f"Epochs completed: {epochs_completed}")

        # Key metrics
        metrics = {
            'mAP50': 'metrics/mAP50(B)',
            'Precision': 'metrics/precision(B)', 
            'Recall': 'metrics/recall(B)',
            'mAP50-95': 'metrics/mAP50-95(B)'
        }

        # Print final values and improvements
        for name, col in metrics.items():
            if col in df.columns:
                final_val = df[col].iloc[-1]
                max_val = df[col].max()
                improvement = final_val - df[col].iloc[0] if len(df) > 1 else 0

                print(f"{name}: {final_val:.4f} (Max: {max_val:.4f}, Δ: {improvement:+.4f})")

        # Loss analysis if available
        loss_cols = [col for col in df.columns if 'loss' in col.lower()]
        if loss_cols:
            print(f"\nLOSS ANALYSIS:")
            for loss_col in loss_cols:
                final_loss = df[loss_col].iloc[-1]
                min_loss = df[loss_col].min()
                print(f"{loss_col}: {final_loss:.4f} (Min: {min_loss:.4f})")

        # Create comprehensive plots
        self._create_yolo_plots(df, mode, save_plots, output_dir)

        # Training insights
        self._generate_insights(df, mode)
    
    def _analyze_custom_metrics(self, df, mode, save_plots, output_dir):
        """Analyze custom detection/segmentation metrics"""
        print(f"\nCUSTOM {mode.upper()} METRICS")
        print("-" * 50)

        latest = df.iloc[-1]
        print(f"Timestamp: {latest.get('timestamp', 'N/A')}")
        print(f"Total Images: {latest.get('total_images', 'N/A')}")
        print(f"Processed: {latest.get('processed_images', 'N/A')}")
        print(f"Failed: {latest.get('failed_images', 'N/A')}")

        if 'failed_images' in df.columns and 'total_images' in df.columns:
            failure_rate = (latest['failed_images'] / latest['total_images']) * 100
            print(f"Failure Rate: {failure_rate:.2f}%")

        # Class distribution analysis
        class_cols = [c for c in df.columns if c not in 
                     ['timestamp', 'total_images', 'processed_images', 
                      'failed_images', 'total_detections', 'total_segments']]

        if class_cols:
            self._create_class_distribution_plot(df, class_cols, mode, save_plots, output_dir)

            # Class statistics
            class_data = latest[class_cols]
            print(f"\nCLASS STATISTICS:")
            print(f"Total classes: {len(class_cols)}")
            print(f"Most frequent: {class_data.idxmax()} ({class_data.max()} instances)")
            print(f"Least frequent: {class_data.idxmin()} ({class_data.min()} instances)")

            # Class imbalance analysis
            imbalance_ratio = class_data.max() / class_data.min() if class_data.min() > 0 else float('inf')
            if imbalance_ratio > 10:
                print(f"High class imbalance detected: {imbalance_ratio:.1f}x ratio")
    
    def _create_yolo_plots(self, df, mode, save_plots, output_dir):
        """Create comprehensive plots for YOLO metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'YOLOv8 {mode.capitalize()} Training Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Key metrics
        if 'metrics/mAP50(B)' in df.columns:
            axes[0, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', linewidth=2, color=self.colors[0])
            axes[0, 0].set_title('mAP50 Evolution', fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('mAP50')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
        
        # Plot 2: Precision-Recall
        if all(col in df.columns for col in ['metrics/precision(B)', 'metrics/recall(B)']):
            axes[0, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2, color=self.colors[1])
            axes[0, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2, color=self.colors[2])
            axes[0, 1].set_title('Precision & Recall', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # Plot 3: Loss curves
        loss_cols = [col for col in df.columns if 'loss' in col.lower() and 'val' not in col.lower()]
        for i, loss_col in enumerate(loss_cols[:3]):  # Plot max 3 loss types
            axes[1, 0].plot(df['epoch'], df[loss_col], label=loss_col, linewidth=2, 
                           color=self.colors[3 + i], alpha=0.8)
        axes[1, 0].set_title('Training Losses', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Learning rate (if available)
        lr_cols = [col for col in df.columns if 'lr' in col.lower() or 'learning_rate' in col.lower()]
        if lr_cols:
            axes[1, 1].plot(df['epoch'], df[lr_cols[0]], label='Learning Rate', linewidth=2, color=self.colors[6])
            axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No Learning Rate Data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(output_dir, f'{mode}_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")
        
        plt.show()
    
    def _create_class_distribution_plot(self, df, class_cols, mode, save_plots, output_dir):
        """Create enhanced class distribution visualization"""
        latest_data = df[class_cols].iloc[-1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{mode.capitalize()} Class Distribution Analysis', fontweight='bold')
        
        # Bar plot
        bars = ax1.bar(range(len(class_cols)), latest_data.values, color=self.colors[:len(class_cols)])
        ax1.set_title('Class Distribution')
        ax1.set_xlabel('Class Index')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Pie chart for proportion
        if latest_data.sum() > 0:
            ax2.pie(latest_data.values, labels=class_cols, autopct='%1.1f%%', 
                   startangle=90, colors=self.colors[:len(class_cols)])
            ax2.set_title('Class Proportion')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(output_dir, f'{mode}_class_distribution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to: {plot_path}")
        
        plt.show()

    def _generate_insights(self, df, mode):
        """Generate training insights and recommendations"""
        print(f"\nTRAINING INSIGHTS:")
        print("-" * 30)

        insights = []

        # Check for convergence
        if 'metrics/mAP50(B)' in df.columns:
            map_data = df['metrics/mAP50(B)']
            last_5_improvement = (map_data.iloc[-1] - map_data.iloc[-5]) if len(map_data) >= 5 else 0

            if last_5_improvement < 0.01:
                insights.append("Model appears to be converging (minimal improvement in last 5 epochs)")
            else:
                insights.append("Model may benefit from more training epochs")

        # Check for overfitting
        train_loss_cols = [col for col in df.columns if 'loss' in col.lower() and 'val' not in col.lower()]
        val_loss_cols = [col for col in df.columns if 'val_loss' in col.lower()]

        if train_loss_cols and val_loss_cols:
            train_loss_final = df[train_loss_cols[0]].iloc[-1]
            val_loss_final = df[val_loss_cols[0]].iloc[-1]

            if val_loss_final > train_loss_final * 1.2:
                insights.append("Possible overfitting detected (validation loss significantly higher than training)")

        # Print insights
        for insight in insights:
            print(f"- {insight}")

        if not insights:
            print("- No specific insights available - consider checking individual metric trends")

    def _basic_analysis(self, df, mode, save_plots, output_dir):
        """Basic analysis for unrecognized CSV structures"""
        print(f"\nBASIC ANALYSIS FOR UNRECOGNIZED FORMAT")
        print("-" * 50)

        print(f"Columns available: {list(df.columns)}")
        print(f"Data shape: {df.shape}")
        print(f"Data types:\n{df.dtypes}")

        # Plot all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(1, min(3, len(numeric_cols)), figsize=(15, 5))
            if len(numeric_cols) == 1:
                axes = [axes]

            for i, col in enumerate(numeric_cols[:3]):
                axes[i].plot(df.index, df[col], marker='o', linewidth=2)
                axes[i].set_title(f'{col} Trend')
                axes[i].set_xlabel('Index')
                axes[i].set_ylabel(col)
                axes[i].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_plots:
                plot_path = os.path.join(output_dir, f'{mode}_basic_analysis.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Basic analysis plot saved to: {plot_path}")

            plt.show()


def main():
    """Main function with enhanced analysis options"""
    analyzer = YOLOAnalyzer(style='seaborn-v0_8')
    
    print("Starting Enhanced YOLOv8 Metrics Analysis...")
    
    # Define analysis paths
    analysis_paths = [
        ("runs/detect/train/results.csv", "detection"),
        ("runs/segment/train/results.csv", "segmentation"),
        # Add more paths as needed
    ]
    
    for path, mode in analysis_paths:
        if os.path.exists(path):
            analyzer.analyze_results(
                path=path,
                mode=mode,
                save_plots=True,  # Set to False if you don't want to save plots
                output_dir=None   # Uses same directory as results file
            )
        else:
            print(f"Results file not found: {path}")
    
    print("\nEnhanced analysis complete! Check the generated plots and insights.")


if __name__ == "__main__":
    main()