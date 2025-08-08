def is_classiq_available():
    """Check if Classiq SDK is installed and importable."""
    try:
        import classiq
        return True
    except ImportError:
        return False

if __name__ == "__main__":
    if is_classiq_available():
        print("[OK] Classiq is installed")
    else:
        print("[X] Classiq not found - install with: pip install classiq")