import subprocess
import os
import sys
import shutil
from pathlib import Path

specfile = 'gpxmapper.spec'

def main():
    """Build a standalone executable for the gpxmapper application."""
    print("Installing PyInstaller...")
    try:
        # Try using pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    except Exception as e:
        print(f"Failed to install using pip: {e}")
        # Try using uv if pip fails
        try:
            print("Attempting to install using uv...")
            subprocess.check_call(["uv", "pip", "install", "pyinstaller"])
        except Exception as e2:
            print(f"Failed to install using uv: {e2}")
            print("Please install PyInstaller manually with: pip install pyinstaller")
            print("Then run this script again.")
            sys.exit(1)

    print("Building executable...")
    # Base PyInstaller command
    pyinstaller_cmd = [
        "pyinstaller",
        "--name=gpxmapper",
        "--onefile",  # Create a single executable file
        "--console",  # Show console window for CLI app
        # Hidden imports for dependencies
        "--hidden-import=gpxpy",
        "--hidden-import=typer",
        "--hidden-import=cv2",  # OpenCV
        "--hidden-import=requests",
        "--hidden-import=PIL",  # Pillow
        "--hidden-import=numpy",
    ]

    # Add LICENSE file if it exists
    license_path = Path("LICENSE")
    if license_path.exists():
        pyinstaller_cmd.append(f"--add-data={license_path};.")

    # Add entry point script
    pyinstaller_cmd.append("exe_entry_point.py")

    # Create a spec file first
    print("Creating spec file...")
    spec_cmd = pyinstaller_cmd.copy()
    spec_cmd.append("--specpath=.")
    spec_cmd.append("--name=gpxmapper")

    # Add --specpath to create the spec file in the current directory
    subprocess.check_call(spec_cmd)

    # Modify the spec file to include additional data files
    print("Modifying spec file...")
    with open(specfile, "r") as f:
        spec_content = f.read()

    # Add datas collection for potential data files
    spec_content = spec_content.replace(
        "datas=[]",
        "datas=[]"  # We'll use the --add-data option for specific files
    )

    # Write the modified spec file
    with open(specfile, "w") as f:
        f.write(spec_content)

    # Run PyInstaller with the spec file
    print("Building executable from spec file...")
    subprocess.check_call(["pyinstaller", specfile])

    print("Executable built successfully!")
    print("You can find it in the 'dist' directory.")

    # Clean up temporary files
    print("Cleaning up...")
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")

if __name__ == "__main__":
    main()
