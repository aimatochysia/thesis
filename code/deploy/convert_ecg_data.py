"""
ECG Data Conversion Script

This script converts raw ECG sample data into a usable CSV dataset.

DATA SOURCE REQUIREMENTS:
- Uses ONLY the data located in the 'sample/' directory
- Input: One signal file (100.csv) with MLII and V5 leads
- Input: One annotation file (100annotations.txt)
- IGNORES the V5 lead completely
- Uses ONLY the MLII signal

OUTPUT:
- A CSV file containing only the MLII signal values (one column per row/time frame)

Usage:
    python convert_ecg_data.py
"""

import pandas as pd
import os


def parse_mlii_signal(signal_csv_path: str) -> pd.DataFrame:
    """
    Parse the MLII signal from the signal CSV file.
    
    Args:
        signal_csv_path: Path to the signal CSV file (e.g., 100.csv)
        
    Returns:
        DataFrame with sample index and MLII values only
    """
    # Read the signal file
    df = pd.read_csv(signal_csv_path)
    
    # Strip quotes from column names if present
    df.columns = df.columns.str.strip().str.strip("'")
    
    # Extract only the MLII column (ignore V5)
    # The file has columns: 'sample #', 'MLII', 'V5'
    mlii_df = df[['sample #', 'MLII']].copy()
    mlii_df.columns = ['sample_index', 'MLII']
    
    return mlii_df


def parse_annotations(annotation_file_path: str) -> pd.DataFrame:
    """
    Parse the annotation file to extract R-peak locations and beat types.
    
    Args:
        annotation_file_path: Path to the annotation file (e.g., 100annotations.txt)
        
    Returns:
        DataFrame with sample index and beat type
    """
    annotations = []
    
    with open(annotation_file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header line
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 4:
            # Format: Time  Sample #  Type  Sub Chan  Num  Aux
            # E.g.: 0:00.214  77  N  0  0  0
            try:
                sample_idx = int(parts[1])
                beat_type = parts[2]
                annotations.append({
                    'sample_index': sample_idx,
                    'beat_type': beat_type
                })
            except (ValueError, IndexError):
                continue
    
    return pd.DataFrame(annotations)


def convert_to_usable_csv(mlii_df: pd.DataFrame, output_path: str) -> None:
    """
    Convert the MLII signal DataFrame to a usable CSV format.
    
    The output CSV contains:
    - Each ROW represents a time frame
    - Each COLUMN represents a signal channel
    - Since only MLII is used, there is exactly ONE data column
    
    Args:
        mlii_df: DataFrame with MLII signal values
        output_path: Path to save the output CSV
    """
    # Create output with just the MLII column for downstream use
    output_df = mlii_df[['MLII']].copy()
    output_df.to_csv(output_path, index=True, index_label='sample_index')
    print(f"Saved MLII signal to: {output_path}")
    print(f"Total samples: {len(output_df)}")


def main():
    """Main function to convert ECG data."""
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, 'sample')
    
    signal_path = os.path.join(sample_dir, '100.csv')
    annotation_path = os.path.join(sample_dir, '100annotations.txt')
    output_path = os.path.join(sample_dir, 'mlii_signal.csv')
    annotations_output_path = os.path.join(sample_dir, 'parsed_annotations.csv')
    
    # Parse MLII signal
    print("=" * 60)
    print("ECG Data Conversion")
    print("=" * 60)
    print(f"\nReading signal data from: {signal_path}")
    mlii_df = parse_mlii_signal(signal_path)
    
    # Parse annotations
    print(f"Reading annotations from: {annotation_path}")
    annotations_df = parse_annotations(annotation_path)
    
    # Convert to usable CSV format
    print("\nConverting to usable CSV format...")
    convert_to_usable_csv(mlii_df, output_path)
    
    # Also save parsed annotations for reference
    annotations_df.to_csv(annotations_output_path, index=False)
    print(f"Saved parsed annotations to: {annotations_output_path}")
    print(f"Total annotated beats: {len(annotations_df)}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print("MLII Signal:")
    print(f"  - Total samples: {len(mlii_df)}")
    print(f"  - Min value: {mlii_df['MLII'].min()}")
    print(f"  - Max value: {mlii_df['MLII'].max()}")
    print(f"  - Mean value: {mlii_df['MLII'].mean():.2f}")
    
    print("\nAnnotations:")
    print(f"  - Total beats: {len(annotations_df)}")
    beat_type_counts = annotations_df['beat_type'].value_counts()
    for beat_type, count in beat_type_counts.items():
        print(f"  - Type '{beat_type}': {count}")
    
    print("\nOutput files:")
    print(f"  1. {output_path}")
    print(f"  2. {annotations_output_path}")


if __name__ == "__main__":
    main()
