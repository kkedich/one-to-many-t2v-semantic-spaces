
import setuptools

setuptools.setup(
    name="vidutils",
    version="0.0.1",
    author="Karina Bogdan",
    author_email="karina.bogdan@gmail.com",
    description="A package with TF based video utilities",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
