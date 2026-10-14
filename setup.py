from setuptools import setup, find_packages

setup(
    name='lito',
    version='1.2.2',
    description='An AI built entirely from scratch - no pretrained models',
    author='Lito',
    packages=find_packages(),
    py_modules=['main', 'train', 'server'],
    include_package_data=True,
    package_data={'': ['extensions/*.py']},
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
    ],
    entry_points={
        'console_scripts': [
            'lito=main:main',
            'lito-train=train:main',
            'lito-server=server:main',
        ],
    },
)
