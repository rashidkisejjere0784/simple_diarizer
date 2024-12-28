import setuptools

from simple_diarizer import __version__

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding='utf-16') as f:
    install_requires = f.readlines()

setuptools.setup(
    version=__version__,
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
