#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Setup script for OAI BERT Fine-tuning project."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='bert-telecom-finetuning',
    version='1.0.0',
    description='BERT Fine-tuning for Telecommunication Domain with Word-Level Contrastive Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Haihang Jiang',
    author_email='jhh1_swjtu@163.com',
    url='https://github.com/KelvinJiang2333/OAI_BERT_finetuning',
    license='MIT',
    packages=find_packages(exclude=['tests', 'docs', 'examples']),
    install_requires=requirements,
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='bert, nlp, contrastive-learning, fine-tuning, telecommunication, word-level, deep-learning, pytorch',
    project_urls={
        'Bug Reports': 'https://github.com/KelvinJiang2333/OAI_BERT_finetuning/issues',
        'Source': 'https://github.com/KelvinJiang2333/OAI_BERT_finetuning',
        'Documentation': 'https://github.com/KelvinJiang2333/OAI_BERT_finetuning/blob/main/README.md',
    },
)

