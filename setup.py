from setuptools import setup, find_packages

setup(
    name = "ppmpy",
    version = "1.0",
    packages = find_packages(),
    install_requires = ["numpy", "setuptools","math","nugridpy","matplotlib","scipy"],
    author = "Falk Herwig, Sam Jones, Robert Andrassy, Daniel Alexander Bertolino Conti ",
    author_email = "fherwig@uvic.ca")
