from setuptools import setup, find_packages

setup(
    name="h-sitk-reader",
    version="0.1.0",
    description="Medical image reader utilities (SimpleITK / DICOM / NIfTI)",
    long_description=open("README.md", "r", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "pydicom>=3.0",
        "SimpleITK>=2.4",
    ],
    include_package_data=True,
    zip_safe=False,
)

