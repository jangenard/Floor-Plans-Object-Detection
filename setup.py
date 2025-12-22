from setuptools import setup, find_packages

setup(
    name="floorplan_detection",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow>=2.15,<2.17",
        "segmentation-models==1.0.1",
        "requests",
        "numpy",
        "opencv-python",
        "albumentations",
        "matplotlib",
        "scipy",
        "setuptools"
    ],
)