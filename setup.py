from pathlib import Path

from distutils.core import setup
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
    author="David Brochart",
    author_email="david.brochart@gmail.com",
    packages=["pydro"],
    install_requires=["cython", "numpy"],
    ext_modules=cythonize("pydro/pydro.pyx"),
    include_dirs=[numpy.get_include()],
)
