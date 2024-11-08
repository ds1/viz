import os
import re
from pathlib import Path

def update_file_content(file_path, old_import="types", new_import="custom_types"):
    """Update imports in a single file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update import statements
    patterns = [
        (r'from (?:src\.)?types import', f'from src.{new_import} import'),
        (r'from \.types import', f'from .{new_import} import'),
        (r'from ..custom_types import', f'from ..{new_import} import'),
        (r'import custom_types', f'import {new_import}'),
    ]
    
    for old_pattern, new_pattern in patterns:
        content = re.sub(old_pattern, new_pattern, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def rename_and_update_imports(src_dir):
    """Rename types.py and update all imports in the project"""
    # Find types.py
    types_path = Path(src_dir) / 'types.py'
    if not types_path.exists():
        print("types.py not found in the specified directory")
        return
    
    # Rename types.py to custom_types.py
    custom_types_path = types_path.parent / 'custom_types.py'
    try:
        types_path.rename(custom_types_path)
        print(f"Renamed {types_path} to {custom_types_path}")
    except Exception as e:
        print(f"Error renaming file: {e}")
        return
    
    # Update imports in all Python files
    python_files = Path(src_dir).rglob("*.py")
    for file_path in python_files:
        if file_path.name != 'custom_types.py':  # Skip the renamed file
            try:
                update_file_content(file_path)
                print(f"Updated imports in {file_path}")
            except Exception as e:
                print(f"Error updating {file_path}: {e}")

if __name__ == "__main__":
    # Get the src directory from the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'scripts' else script_dir
    
    print(f"Starting update process in {src_dir}")
    rename_and_update_imports(src_dir)
    print("Update complete!")
