import os
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_data(source_dir: str = 'FileTypeData', 
               train_ratio: float = 0.7,
               val_ratio: float = 0.15,
               test_ratio: float = 0.15):
    """
    Split data into train, validation, and test sets.
    
    Args:
        source_dir: Directory containing language-specific subdirectories
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    # Create output directories
    base_dir = Path(source_dir)
    for split in ['train', 'validation', 'test']:
        (base_dir / split).mkdir(exist_ok=True)
    
    # Process each language directory
    for lang_dir in base_dir.iterdir():
        if not lang_dir.is_dir() or lang_dir.name in ['train', 'validation', 'test']:
            continue
            
        print(f"\nProcessing {lang_dir.name}...")
        
        # Get all files
        files = list(lang_dir.glob('*'))
        if not files:
            print(f"No files found in {lang_dir.name}")
            continue
            
        # Split files
        train_files, temp_files = train_test_split(
            files, 
            train_size=train_ratio,
            random_state=42
        )
        
        # Split remaining files into validation and test
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_files, test_files = train_test_split(
            temp_files,
            train_size=val_ratio_adjusted,
            random_state=42
        )
        
        # Create language directories in each split
        for split in ['train', 'validation', 'test']:
            (base_dir / split / lang_dir.name).mkdir(exist_ok=True)
        
        # Copy files to their respective directories
        for split, files in [
            ('train', train_files),
            ('validation', val_files),
            ('test', test_files)
        ]:
            for file in files:
                dest = base_dir / split / lang_dir.name / file.name
                shutil.copy2(file, dest)
            
            print(f"  {split}: {len(files)} files")
    
    print("\nData split complete!")

if __name__ == "__main__":
    split_data() 