from setuptools import setup
from Cython.Build import cythonize
import os

cython_sources = [
    os.path.join("src", "reservoir_sampler.pyx"),
    os.path.join("src", "count_db_items.pyx")
]

setup(
    name="hist_w2v",
    ext_modules=cythonize(cython_sources),
    packages=["hist_w2v"],
)
