import os
import site
import sys
from pathlib import Path

def add_project_to_path():
    # Get the project root (two directories up from this script)
    project_root = Path(__file__).parent.parent.absolute()
    
    # Get the site-packages directory for the current conda environment
    site_packages = site.getsitepackages()[0]
    
    # Create a .pth file in site-packages
    pth_file = os.path.join(site_packages, 'petal_viz.pth')
    
    try:
        with open(pth_file, 'w') as f:
            f.write(str(project_root))
        print(f"Successfully added {project_root} to Python path")
    except Exception as e:
        print(f"Error creating .pth file: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if add_project_to_path():
        print("Setup complete! You can now run your application using:")
        print("python src/main.py")
