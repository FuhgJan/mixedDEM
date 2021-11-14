#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages, Extension

with open('README.md') as readme_file:
    readme = readme_file.read()

test_requirements = ['pytest>=3', ]
setup_requirements = ['pytest-runner', ]

setup(
    author="Jan N. Fuhg",
    python_requires='>=3',
    classifiers=[
        'Development Status :: 3 - Beta',
        'Intended Audience :: Experts',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="mixedDEM library",
    install_requires=['numpy', 'scipy', 'torch','matplotlib', 'pyevtk', 'triangle'],
    license="GNU General Public License v3",
    long_description=readme,
    name='mDEM',
)
