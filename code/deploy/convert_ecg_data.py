import pandas as pd
import os


def parse_mlii_signal(signal_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(signal_csv_path)
    
    df.columns = df.columns.str.strip().str.strip("'")
    
    mlii_df = df[['sample #', 'MLII']].copy()
    mlii_df.columns = ['sample_index', 'MLII']
    
    return mlii_df


def parse_annotations(annotation_file_path: str) -> pd.DataFrame:
    annotations = []
    
    with open(annotation_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 4:
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
    output_df = mlii_df[['MLII']].copy()
    output_df.to_csv(output_path, index=True, index_label='sample_index')
    print(f"Saved MLII signal to: {output_path}")
    print(f"Total samples: {len(output_df)}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, 'sample')
    
    signal_path = os.path.join(sample_dir, '100.csv')
    annotation_path = os.path.join(sample_dir, '100annotations.txt')
    output_path = os.path.join(sample_dir, 'mlii_signal.csv')
    annotations_output_path = os.path.join(sample_dir, 'parsed_annotations.csv')
    
    print("=" * 60)
    print("ECG Data Conversion")
    print("=" * 60)
    print(f"\nReading signal data from: {signal_path}")
    mlii_df = parse_mlii_signal(signal_path)
    
    print(f"Reading annotations from: {annotation_path}")
    annotations_df = parse_annotations(annotation_path)
    
    print("\nConverting to usable CSV format...")
    convert_to_usable_csv(mlii_df, output_path)
    
    annotations_df.to_csv(annotations_output_path, index=False)
    print(f"Saved parsed annotations to: {annotations_output_path}")
    print(f"Total annotated beats: {len(annotations_df)}")
    
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
