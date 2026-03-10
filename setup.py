from setuptools import setup, find_packages

setup(
    name="mosquito_yolo",
    version="1.0",
    description="Mosquito Detection using YOLOv12",
    author="MSCS Research",
    packages=find_packages(),
    install_requires=[
        "ultralytics",
        "opencv-python",
        "numpy",
        "matplotlib",
        "pyyaml"
    ]
)
