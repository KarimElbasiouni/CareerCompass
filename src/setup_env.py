import sys

def check_versions():
    print("[info] Python version:", sys.version)

    try:
        import pandas as pd
        print("[ok] pandas", pd.__version__)
    except ImportError:
        print("[error] pandas not installed")

    try:
        import sklearn
        print("[ok] scikit-learn", sklearn.__version__)
    except ImportError:
        print("[warn] scikit-learn not installed (you’ll need it later)")

if __name__ == "__main__":
    check_versions()
