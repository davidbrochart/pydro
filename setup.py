from pathlib import Path

from setuptools import setup
from Cython.Build import cythonize
import numpy


HERE = Path()
code = (HERE / "pydro" / "_version.py").read_text()
namespace = {}
exec(code, {}, namespace)

setup(
    name="pydro",
    version=namespace["version"],
    description="Hydrological modelling in Python",
    long_description=(HERE / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="David Brochart",
    author_email="david.brochart@gmail.com",
    packages=["pydro"],
    ext_modules=cythonize("pydro/pydro.pyx"),
    include_dirs=[numpy.get_include()],
)
