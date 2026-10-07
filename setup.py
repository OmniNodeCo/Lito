from setuptools import setup, find_packages

setup(
    name='smartai',
    version='1.0.0',
    description='An AI built entirely from scratch - no pretrained models',
    author='SmartAI',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
    ],
    entry_points={
        'console_scripts': [
            'smartai=main:main',
            'smartai-train=train:main',
        ],
    },
)