import platform
import subprocess
import sys


def build_exe():
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pyinstaller"]
        )

    separator = ";" if platform.system() == "Windows" else ":"
    data_argument = f"data.json{separator}."

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

    print("Building executable...")
    print(" ".join(pyinstaller_args))
    print()

    result = subprocess.run(pyinstaller_args)

    if result.returncode == 0:
        print("\n" + "=" * 45)
        print("BUILD SUCCESSFUL!")
        print("Your .exe is in the 'dist/' folder.")
        print("Note: lito_memory.db will be created next to")
        print("the .exe on first run (it cannot be baked in).")
        print("=" * 45)
    else:
        print("\nBuild failed. Check the error output above.")


if __name__ == "__main__":
    build_exe()