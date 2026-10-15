
import setuptools

setuptools.setup(
    name="vqa",
    version="0.0.1",
    author="Karina Bogdan",
    author_email="karina.bogdan@gmail.com",
    description="A package to generate features and other related requirements for the VQA metrics",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
