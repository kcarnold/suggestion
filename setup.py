try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('suggestion.suffix_sort', ['suggestion/qsufsort.c', 'suggestion/suffix_sort.pyx'], include_dirs=[np.get_include()])
]

setup(
    name="suggestion",
    packages=['suggestion'],
    ext_modules=cythonize(extensions)
)
