import os
import platform
import subprocess
import sys


def build_exe():
    # 1. Check if model exists, train if not
    if not os.path.exists("lito_model.pkl"):
        print("Model file not found. Running train.py...")
        train_res = subprocess.run([sys.executable, "train.py"])
        if train_res.returncode != 0:
            print("Error: train.py failed to run!")
            sys.exit(1)

    # 2. Find the responses JSON file (handles either name)
    json_file = "responses.json"
    if not os.path.exists(json_file):
        if os.path.exists("data.json"):
            json_file = "data.json"
        else:
            print("Error: Neither 'responses.json' nor 'data.json' found in repo!")
            sys.exit(1)

    separator = ";" if platform.system() == "Windows" else ":"

    # 3. Build arguments with scikit-learn full collection
    pyinstaller_args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--console",
        "--name",
        "LitoBot",
        "--collect-all",
        "sklearn",
        "--collect-all",
        "joblib",
        "--add-data",
        f"lito_model.pkl{separator}.",
        "--add-data",
        f"{json_file}{separator}.",
        "main.py",
    ]

    print("\nRunning PyInstaller build command:")
    print(" ".join(pyinstaller_args))
    print("=" * 50)

    # 4. Run PyInstaller
    result = subprocess.run(pyinstaller_args)

    if result.returncode != 0:
        print(f"\n[ERROR] PyInstaller failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    # 5. Verify the exe was actually produced
    exe_name = "LitoBot.exe" if platform.system() == "Windows" else "LitoBot"
    expected_exe = os.path.join("dist", exe_name)
    if not os.path.exists(expected_exe):
        print(f"\n[ERROR] Build finished but '{expected_exe}' was not found!")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("BUILD SUCCESSFUL!")
    print(f"File created: {expected_exe}")
    print("=" * 50)


if __name__ == "__main__":
    build_exe()