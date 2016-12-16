try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    # Extension('orgback.accel', ['orgback/accel.pyx'], include_dirs=[np.get_include()])
]

setup(
    name="suggestion",
    packages=['suggestion'],
    ext_modules=cythonize(extensions)
)
