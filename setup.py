#!/usr/bin/env python
# coding: utf8
import setuptools
from os import path



## Open README
here = path.abspath(path.dirname(__file__))
readme_path = path.join(here, 'README.md')
with open(readme_path, 'r') as f:
    readme = f.read()

setuptools.setup(
            name='wNMF',
            version='0.0.1',
            long_description=readme,
            description='wNMF: weighted Non-Negative matrix Factorization',
            long_description_content_type='text/markdown',
            author='SN',
            author_email='scottnanda@gmail.com',
            url='https://github.com/asn32/weighted-nmf',
            license='MIT License',
            packages=['wNMF'],
            python_requires='>=3.6',
            install_requires='numpy>=1.13',
            include_package_data=False,
            classifiers=[
                "Programming Language :: Python",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.6",
                "Topic :: Scientific/Engineering :: Bio-Informatics",
                "Topic :: Scientific/Engineering :: Mathematics"
            ]
)