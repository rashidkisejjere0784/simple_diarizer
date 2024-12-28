import os

__version__ = os.getenv("GITHUB_REF_NAME", "0.0.1")
