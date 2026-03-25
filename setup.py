from setuptools import find_packages, setup

setup(
    name="medcore",
    version="1.2.0",
    description="Medical imaging utilities based on SimpleITK",
    long_description=open("README.md", "r", encoding="utf-8").read()
    if __import__("os").path.exists("README.md")
    else "",
    long_description_content_type="text/markdown",
    author="yongho choi",
    author_email="yhchoi@hutom.co.kr",
    maintainer="DATA",
    maintainer_email="dm@hutom.co.kr",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "pydicom>=3.0",
        "SimpleITK>=2.4",
        "matplotlib>=3.7",
        "opencv-python-headless>=4.8",
        "scikit-image>=0.21",
        "scipy>=1.11",
    ],
    extras_require={
        "dev": [
            "pre-commit>=4.0",
            "pytest>=8.0",
            "ruff>=0.11.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
