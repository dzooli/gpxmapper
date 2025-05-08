import subprocess
import os
import sys
import shutil
from pathlib import Path

# Try to import tomli for TOML parsing
try:
    import tomli
except ImportError:
    # If tomli is not available, try to install it using uv or pip
    print("tomli module not found. Attempting to install...")
    try:
        subprocess.check_call(["uv", "pip", "install", "tomli"])
        import tomli
    except Exception as e:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tomli"])
            import tomli
        except Exception as e2:
            print(f"Failed to install tomli: {e2}")
            print("Continuing without TOML parsing support.")
            tomli = None

SPECFILE = 'gpxmapper.spec'
SPECPATH_OPT = '--specpath=.'


def main():
    """Build a standalone executable for the gpxmapper application."""
    install_pyinstaller()
    config = load_config()
    build_executable(config)
    cleanup_temp_files()


def install_pyinstaller():
    """Install PyInstaller using uv or pip."""
    print("Installing PyInstaller...")
    try:
        print("Attempting to install using uv...")
        subprocess.check_call(["uv", "pip", "install", "pyinstaller"])
        return
    except Exception as e:
        print(f"Failed to install using uv: {e}")

    try:
        print("Falling back to pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        return
    except Exception as e:
        print(f"Failed to install using pip: {e}")
        print("Please install PyInstaller manually with: pip install pyinstaller")
        print("Then run this script again.")
        sys.exit(1)


def load_config():
    """Load PyInstaller configuration from pyproject.toml."""
    if tomli is None:
        print("TOML parsing not available. Using default PyInstaller configuration.")
        return {}

    try:
        with open("pyproject.toml", "rb") as f:
            pyproject_data = tomli.load(f)
        config = pyproject_data.get("tool", {}).get("pyinstaller", {})
        if not config:
            print("No PyInstaller configuration found in pyproject.toml, using defaults.")
        return config
    except Exception as e:
        print(f"Failed to load configuration from pyproject.toml: {e}")
        print("Using default PyInstaller configuration.")
        return {}


def get_pyinstaller_command(config):
    """Build PyInstaller command with options."""
    cmd = ["pyinstaller"]

    if config and "options" in config:
        cmd.extend(config["options"])
    else:
        cmd.extend([
            "--name=gpxmapper",
            "--onefile",
            "--console",
            "--hidden-import=gpxpy",
            "--hidden-import=typer",
            "--hidden-import=cv2",
            "--hidden-import=requests",
            "--hidden-import=PIL",
            "--hidden-import=numpy",
            SPECPATH_OPT,
        ])

    # Add LICENSE file if exists
    if Path("LICENSE").exists():
        cmd.append("--add-data=LICENSE;.")

    # Add entry point
    entry_point = config.get("entry_point", "exe_entry_point.py")
    cmd.append(entry_point)

    return cmd


def create_spec_file(pyinstaller_cmd):
    """Create and modify spec file."""
    spec_cmd = pyinstaller_cmd.copy()

    if SPECPATH_OPT not in spec_cmd:
        spec_cmd.append(SPECPATH_OPT)
    if not any(opt.startswith("--name=") for opt in spec_cmd):
        spec_cmd.append("--name=gpxmapper")

    print(f"Running command: {' '.join(spec_cmd)}")
    subprocess.check_call(spec_cmd)

    # Modify spec file
    print("Modifying spec file...")
    with open(SPECFILE, "r") as f:
        spec_content = f.read()

    spec_content = spec_content.replace("datas=[]", "datas=[]")

    with open(SPECFILE, "w") as f:
        f.write(spec_content)


def build_from_spec():
    """Build executable from spec file."""
    print("Building executable from spec file...")
    try:
        print("Using uv to build executable...")
        subprocess.check_call(["uv", "run", "pyinstaller", SPECFILE])
    except Exception as e:
        print(f"Failed to build using uv: {e}")
        print("Falling back to direct PyInstaller call...")
        subprocess.check_call(["pyinstaller", SPECFILE])


def build_executable(config):
    """Build the executable using PyInstaller."""
    print("Building executable...")
    pyinstaller_cmd = get_pyinstaller_command(config)
    create_spec_file(pyinstaller_cmd)
    build_from_spec()
    print("Executable built successfully!")
    print("You can find it in the 'dist' directory.")


def cleanup_temp_files():
    """Clean up temporary build files."""
    print("Cleaning up...")
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")


if __name__ == "__main__":
    main()
