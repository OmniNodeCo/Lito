import os
import platform
import subprocess
import sys


def build_exe():
    # Ensure dependencies are installed
    for pkg in ["pyinstaller", "scikit-learn", "joblib"]:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg]
            )

    # Check if model exists, train if not
    if not os.path.exists("lito_model.pkl"):
        print("No model found. Training first...")
        subprocess.check_call([sys.executable, "train.py"])

    separator = ";" if platform.system() == "Windows" else ":"

    pyinstaller_args = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
        "--console",
        "--name", "LitoBot",
        "--add-data", f"lito_model.pkl{separator}.",
        "--add-data", f"responses.json{separator}.",
        "main.py",
    ]

    print("\nBuilding executable...")
    print(" ".join(pyinstaller_args))
    print()

    result = subprocess.run(pyinstaller_args)

    if result.returncode == 0:
        print("\n" + "=" * 50)
        print("BUILD SUCCESSFUL!")
        print("Your .exe is in the 'dist/' folder.")
        print()
        print("Note: lito_memory.db will be created next to")
        print("the .exe on first run (databases must be writable).")
        print("=" * 50)
    else:
        print("\nBuild failed. Check the error output above.")


if __name__ == "__main__":
    build_exe()