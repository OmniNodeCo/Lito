import platform
import subprocess
import sys


def build_exe():
    # 1. Ensure PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing it now...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pyinstaller"]
        )

    # 2. Determine OS separator for data files (';' on Windows, ':' on Mac/Linux)
    separator = ";" if platform.system() == "Windows" else ":"
    data_argument = f"data.json{separator}."

    # 3. Build arguments
    pyinstaller_args = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
        "--console",
        "--name",
        "LitoBot",
        "--add-data",
        data_argument,
        "main.py",
    ]

    print("Building executable with command:")
    print(" ".join(pyinstaller_args))
    print("\nStarting build process...\n")

    # 4. Run the build
    result = subprocess.run(pyinstaller_args)

    if result.returncode == 0:
        print("\n" + "=" * 45)
        print("BUILD SUCCESSFUL!")
        print("Your executable is located inside the 'dist/' folder.")
        print("=" * 45)
    else:
        print("\nBuild failed. Check the error output above.")


if __name__ == "__main__":
    build_exe()