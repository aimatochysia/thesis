#!/usr/bin/env python3
"""
Model Architecture Visualization Script
Generates a beautiful diagram of the Context-Aware CNN1D model for thesis visualization.

Usage:
    python visualize_model.py

Output:
    - model_architecture.png (block diagram)
    - model_architecture_detailed.png (detailed with dimensions)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Color scheme for thesis (professional blue theme)
COLORS = {
    'input': '#E3F2FD',       # Light blue
    'conv': '#2196F3',        # Blue
    'bn': '#64B5F6',          # Light blue
    'relu': '#FF9800',        # Orange
    'pool': '#4CAF50',        # Green
    'gap': '#9C27B0',         # Purple
    'fc': '#F44336',          # Red
    'dropout': '#607D8B',     # Gray
    'output': '#E8F5E9',      # Light green
    'arrow': '#37474F',       # Dark gray
    'text': '#212121',        # Almost black
}

def create_block_diagram():
    """Create a clean block diagram of the model architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Title
    ax.text(8, 9.5, 'Context-Aware CNN1D Architecture', 
            fontsize=18, fontweight='bold', ha='center', va='center',
            color=COLORS['text'])
    ax.text(8, 9.0, 'for ECG Arrhythmia Detection',
            fontsize=12, ha='center', va='center', color='gray')
    
    # Layer positions (x, y, width, height)
    layers = [
        # Input
        {'name': 'Input\n(7, 200)', 'pos': (0.5, 5), 'size': (1.5, 1.5), 
         'color': COLORS['input'], 'type': 'input'},
        
        # Conv Block 1
        {'name': 'Conv1D\n7→16\nk=3', 'pos': (2.5, 5), 'size': (1.2, 1.5), 
         'color': COLORS['conv'], 'type': 'conv'},
        {'name': 'BN', 'pos': (3.9, 5.5), 'size': (0.5, 0.5), 
         'color': COLORS['bn'], 'type': 'bn'},
        {'name': 'ReLU', 'pos': (3.9, 4.8), 'size': (0.5, 0.5), 
         'color': COLORS['relu'], 'type': 'relu'},
        {'name': 'MaxPool\n(2)', 'pos': (4.6, 5), 'size': (0.8, 1.5), 
         'color': COLORS['pool'], 'type': 'pool'},
        
        # Conv Block 2
        {'name': 'Conv1D\n16→32\nk=5', 'pos': (6, 5), 'size': (1.2, 1.5), 
         'color': COLORS['conv'], 'type': 'conv'},
        {'name': 'BN', 'pos': (7.4, 5.5), 'size': (0.5, 0.5), 
         'color': COLORS['bn'], 'type': 'bn'},
        {'name': 'ReLU', 'pos': (7.4, 4.8), 'size': (0.5, 0.5), 
         'color': COLORS['relu'], 'type': 'relu'},
        {'name': 'MaxPool\n(2)', 'pos': (8.1, 5), 'size': (0.8, 1.5), 
         'color': COLORS['pool'], 'type': 'pool'},
        
        # Conv Block 3
        {'name': 'Conv1D\n32→64\nk=7', 'pos': (9.5, 5), 'size': (1.2, 1.5), 
         'color': COLORS['conv'], 'type': 'conv'},
        {'name': 'BN', 'pos': (10.9, 5.5), 'size': (0.5, 0.5), 
         'color': COLORS['bn'], 'type': 'bn'},
        {'name': 'ReLU', 'pos': (10.9, 4.8), 'size': (0.5, 0.5), 
         'color': COLORS['relu'], 'type': 'relu'},
        {'name': 'MaxPool\n(2)', 'pos': (11.6, 5), 'size': (0.8, 1.5), 
         'color': COLORS['pool'], 'type': 'pool'},
        
        # Global Average Pooling
        {'name': 'Global\nAvgPool', 'pos': (13, 5), 'size': (1, 1.5), 
         'color': COLORS['gap'], 'type': 'gap'},
        
        # Classifier
        {'name': 'FC\n64→32', 'pos': (14.5, 6.5), 'size': (1, 1), 
         'color': COLORS['fc'], 'type': 'fc'},
        {'name': 'ReLU', 'pos': (14.5, 5.3), 'size': (1, 0.6), 
         'color': COLORS['relu'], 'type': 'relu'},
        {'name': 'Dropout\n(0.5)', 'pos': (14.5, 4.5), 'size': (1, 0.6), 
         'color': COLORS['dropout'], 'type': 'dropout'},
        {'name': 'FC\n32→2', 'pos': (14.5, 3.5), 'size': (1, 1), 
         'color': COLORS['fc'], 'type': 'fc'},
        
        # Output
        {'name': 'Output\n(2)', 'pos': (14.5, 2), 'size': (1, 0.8), 
         'color': COLORS['output'], 'type': 'output'},
    ]
    
    # Draw layers
    for layer in layers:
        x, y = layer['pos']
        w, h = layer['size']
        
        # Draw box
        box = FancyBboxPatch((x, y - h/2), w, h,
                             boxstyle="round,pad=0.03,rounding_size=0.1",
                             facecolor=layer['color'],
                             edgecolor='black',
                             linewidth=1.5)
        ax.add_patch(box)
        
        # Draw text
        text_color = 'white' if layer['type'] in ['conv', 'gap', 'fc', 'dropout'] else COLORS['text']
        ax.text(x + w/2, y, layer['name'],
                fontsize=8, ha='center', va='center',
                fontweight='bold', color=text_color)
    
    # Draw arrows (connections)
    arrow_style = "Simple,tail_width=0.5,head_width=4,head_length=6"
    
    # Horizontal flow arrows
    horizontal_connections = [
        (2.0, 5, 2.5, 5),      # Input → Conv1
        (4.4, 5, 4.6, 5),      # Conv1 block
        (5.4, 5, 6.0, 5),      # Pool1 → Conv2
        (7.9, 5, 8.1, 5),      # Conv2 block
        (8.9, 5, 9.5, 5),      # Pool2 → Conv3
        (11.4, 5, 11.6, 5),    # Conv3 block
        (12.4, 5, 13.0, 5),    # Pool3 → GAP
    ]
    
    for x1, y1, x2, y2 in horizontal_connections:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.5))
    
    # GAP to FC (turn down)
    ax.annotate('', xy=(14.5, 7.0), xytext=(14.0, 5),
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.5,
                               connectionstyle="arc3,rad=-0.3"))
    
    # Vertical flow in classifier
    vertical_connections = [
        (15, 6.0, 15, 5.6),    # FC1 → ReLU
        (15, 5.0, 15, 4.8),    # ReLU → Dropout
        (15, 4.2, 15, 4.0),    # Dropout → FC2
        (15, 3.0, 15, 2.4),    # FC2 → Output
    ]
    
    for x1, y1, x2, y2 in vertical_connections:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.5))
    
    # Add dimension annotations
    dims = [
        (1.25, 3.5, '7×200'),
        (3.1, 3.5, '16×100'),
        (6.6, 3.5, '32×50'),
        (10.1, 3.5, '64×25'),
        (13.5, 3.5, '64'),
    ]
    
    for x, y, text in dims:
        ax.text(x, y, text, fontsize=8, ha='center', va='center',
                color='gray', style='italic')
    
    # Add legend
    legend_items = [
        ('Conv1D', COLORS['conv']),
        ('BatchNorm', COLORS['bn']),
        ('ReLU', COLORS['relu']),
        ('MaxPool', COLORS['pool']),
        ('GlobalAvgPool', COLORS['gap']),
        ('Linear (FC)', COLORS['fc']),
        ('Dropout', COLORS['dropout']),
    ]
    
    legend_y = 1.5
    for i, (name, color) in enumerate(legend_items):
        x = 1 + i * 2
        box = FancyBboxPatch((x, legend_y - 0.2), 0.4, 0.4,
                             boxstyle="round,pad=0.02",
                             facecolor=color,
                             edgecolor='black',
                             linewidth=1)
        ax.add_patch(box)
        ax.text(x + 0.6, legend_y, name, fontsize=8, va='center')
    
    # Add block labels
    ax.text(3.5, 7.5, 'Conv Block 1', fontsize=10, ha='center', fontweight='bold', color='gray')
    ax.text(7.0, 7.5, 'Conv Block 2', fontsize=10, ha='center', fontweight='bold', color='gray')
    ax.text(10.5, 7.5, 'Conv Block 3', fontsize=10, ha='center', fontweight='bold', color='gray')
    ax.text(15, 7.5, 'Classifier', fontsize=10, ha='center', fontweight='bold', color='gray')
    
    # Draw block boundaries
    for x1, x2 in [(2.3, 5.5), (5.8, 9.0), (9.3, 12.5)]:
        rect = plt.Rectangle((x1, 3.8), x2-x1, 4.5, 
                             fill=False, edgecolor='lightgray', 
                             linestyle='--', linewidth=1)
        ax.add_patch(rect)
    
    plt.tight_layout()
    return fig


def create_vertical_diagram():
    """Create a vertical flow diagram suitable for slides."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 14))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(4, 13.5, 'Context-Aware CNN1D', 
            fontsize=16, fontweight='bold', ha='center', va='center')
    
    # Layer definitions (name, y_position, color, dimensions)
    layers = [
        ('Input: Context Window', 12.5, COLORS['input'], '(batch, 7, 200)'),
        ('Conv1D (7→16, k=3) + BN + ReLU', 11.3, COLORS['conv'], ''),
        ('MaxPool1d(2)', 10.5, COLORS['pool'], '→ (batch, 16, 100)'),
        ('Conv1D (16→32, k=5) + BN + ReLU', 9.3, COLORS['conv'], ''),
        ('MaxPool1d(2)', 8.5, COLORS['pool'], '→ (batch, 32, 50)'),
        ('Conv1D (32→64, k=7) + BN + ReLU', 7.3, COLORS['conv'], ''),
        ('MaxPool1d(2)', 6.5, COLORS['pool'], '→ (batch, 64, 25)'),
        ('Global Average Pooling', 5.3, COLORS['gap'], '→ (batch, 64)'),
        ('Linear (64→32) + ReLU', 4.1, COLORS['fc'], ''),
        ('Dropout (0.5)', 3.3, COLORS['dropout'], ''),
        ('Linear (32→2)', 2.5, COLORS['fc'], ''),
        ('Output: [Normal, Abnormal]', 1.3, COLORS['output'], '(batch, 2)'),
    ]
    
    box_width = 5.5
    box_height = 0.6
    
    for name, y, color, dims in layers:
        # Draw box
        x = (8 - box_width) / 2
        box = FancyBboxPatch((x, y - box_height/2), box_width, box_height,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=color,
                             edgecolor='black',
                             linewidth=1.5)
        ax.add_patch(box)
        
        # Text color based on background
        text_color = 'white' if color in [COLORS['conv'], COLORS['gap'], 
                                          COLORS['fc'], COLORS['dropout']] else COLORS['text']
        ax.text(4, y, name, fontsize=10, ha='center', va='center',
                fontweight='bold', color=text_color)
        
        # Dimension annotation
        if dims:
            ax.text(7.2, y, dims, fontsize=8, ha='left', va='center',
                    color='gray', style='italic')
    
    # Draw arrows between layers
    for i in range(len(layers) - 1):
        y1 = layers[i][1] - box_height/2 - 0.05
        y2 = layers[i+1][1] + box_height/2 + 0.05
        ax.annotate('', xy=(4, y2), xytext=(4, y1),
                    arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=2))
    
    # Add parameter count
    ax.text(4, 0.5, 'Total Parameters: ~7,426', 
            fontsize=10, ha='center', va='center', 
            color='gray', style='italic')
    
    plt.tight_layout()
    return fig


def create_simple_flowchart():
    """Create a simple flowchart for quick presentation."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(6, 5.5, 'Model Pipeline', fontsize=14, fontweight='bold', ha='center')
    
    # Boxes
    boxes = [
        ('7-Beat\nContext\nWindow', 1, 3, COLORS['input']),
        ('Conv\nBlocks\n×3', 3.5, 3, COLORS['conv']),
        ('Global\nAvg\nPool', 6, 3, COLORS['gap']),
        ('FC\nLayers', 8.5, 3, COLORS['fc']),
        ('Normal/\nAbnormal', 11, 3, COLORS['output']),
    ]
    
    for name, x, y, color in boxes:
        box = FancyBboxPatch((x-0.8, y-0.8), 1.6, 1.6,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=color,
                             edgecolor='black',
                             linewidth=2)
        ax.add_patch(box)
        
        text_color = 'white' if color in [COLORS['conv'], COLORS['gap'], COLORS['fc']] else 'black'
        ax.text(x, y, name, fontsize=10, ha='center', va='center',
                fontweight='bold', color=text_color)
    
    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][1] + 0.9
        x2 = boxes[i+1][1] - 0.9
        ax.annotate('', xy=(x2, 3), xytext=(x1, 3),
                    arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=2))
    
    # Dimension annotations below
    dims = ['(7, 200)', '16→32→64', '64', '64→32→2', '2 classes']
    for i, dim in enumerate(dims):
        x = boxes[i][1]
        ax.text(x, 1.5, dim, fontsize=9, ha='center', color='gray')
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("Generating model architecture visualizations...")
    
    # Generate all three diagrams
    fig1 = create_block_diagram()
    fig1.savefig('model_architecture_block.png', dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("✓ Saved: model_architecture_block.png")
    
    fig2 = create_vertical_diagram()
    fig2.savefig('model_architecture_vertical.png', dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("✓ Saved: model_architecture_vertical.png")
    
    fig3 = create_simple_flowchart()
    fig3.savefig('model_architecture_simple.png', dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("✓ Saved: model_architecture_simple.png")
    
    print("\nAll visualizations generated successfully!")
    print("\nFiles created:")
    print("  1. model_architecture_block.png   - Detailed horizontal block diagram")
    print("  2. model_architecture_vertical.png - Vertical flow diagram (for slides)")
    print("  3. model_architecture_simple.png  - Simple flowchart overview")
    
    plt.show()
