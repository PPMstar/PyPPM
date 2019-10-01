from setuptools import setup, find_packages

setup(
    name = "ppmpy",
    version = "0.90",
    packages = find_packages(),
    install_requires = ["numpy", "setuptools","nugridpy","matplotlib","scipy"],
    author = "Robert Andrassy,, David Stephens, Falk Herwig, Sam Jones,  Daniel Alexander Bertolino Conti ",
    author_email = "fherwig@uvic.ca")
