import os

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def create_file(path):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            pass  # Create an empty file
        print(f"Created file: {path}")

def create_project_structure():
    base_dir = "petal_viz"
    create_directory(base_dir)

    # Create src directory and its subdirectories
    src_dir = os.path.join(base_dir, "src")
    create_directory(src_dir)
    create_file(os.path.join(src_dir, "main.py"))

    # UI directory
    ui_dir = os.path.join(src_dir, "ui")
    create_directory(ui_dir)
    ui_files = ["main_window.py", "visualizer.py", "timeline.py", "status_bar.py"]
    for file in ui_files:
        create_file(os.path.join(ui_dir, file))

    # Data directory
    data_dir = os.path.join(src_dir, "data")
    create_directory(data_dir)
    data_files = ["lsl_receiver.py", "data_processor.py"]
    for file in data_files:
        create_file(os.path.join(data_dir, file))

    # Utils directory
    utils_dir = os.path.join(src_dir, "utils")
    create_directory(utils_dir)
    utils_files = ["config.py", "constants.py"]
    for file in utils_files:
        create_file(os.path.join(utils_dir, file))

    # Resources directory
    resources_dir = os.path.join(src_dir, "resources")
    create_directory(resources_dir)
    create_directory(os.path.join(resources_dir, "icons"))
    create_directory(os.path.join(resources_dir, "styles"))

    # Tests directory
    create_directory(os.path.join(base_dir, "tests"))

    # Root files
    create_file(os.path.join(base_dir, "requirements.txt"))
    create_file(os.path.join(base_dir, "README.md"))

if __name__ == "__main__":
    create_project_structure()
    print("Project structure created successfully!")