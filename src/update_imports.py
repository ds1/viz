import os
from pathlib import Path

def update_file_imports(file_path):
    """Update imports in a single file from types to custom_types"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified = False
        new_lines = []
        for line in lines:
            if 'from types import' in line:
                # Skip built-in types imports
                new_lines.append(line)
            elif 'import custom_types' in line or 'from src.custom_types import' in line or 'from .custom_types import' in line or 'from ..custom_types import' in line:
                new_line = (
                    line.replace('import custom_types', 'import custom_types')
                    .replace('from src.custom_types import', 'from src.custom_types import')
                    .replace('from .custom_types import', 'from .custom_types import')
                    .replace('from ..custom_types import', 'from ..custom_types import')
                )
                new_lines.append(new_line)
                modified = True
            else:
                new_lines.append(line)

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print(f"Updated imports in {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def update_all_imports(src_dir):
    """Update imports in all Python files in the project"""
    python_files = [
        f for f in Path(src_dir).rglob("*.py")
        if 'custom_types.py' not in str(f)
        and '__pycache__' not in str(f)
    ]
    
    print(f"Found {len(python_files)} Python files to process")
    for file_path in python_files:
        update_file_imports(file_path)

if __name__ == "__main__":
    # Get the current directory
    src_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Starting import updates in {src_dir}")
    update_all_imports(src_dir)
    print("Update complete!")
