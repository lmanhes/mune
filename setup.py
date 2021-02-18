import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pythia",
    version="0.0.1",
    author="Louis Manhès",
    description="Multimodal Universal Explorer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lmanhes/mune",
    packages=setuptools.find_packages(),
    install_requires=["torch==1.7.1"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)