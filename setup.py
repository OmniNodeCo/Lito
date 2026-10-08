from setuptools import setup, find_packages

setup(
    name='lito',
    version='1.0.3',
    description='An AI built entirely from scratch - no pretrained models',
    author='Lito',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
    ],
    entry_points={
        'console_scripts': [
            'lito=main:main',
            'lito-train=train:main',
        ],
    },
)