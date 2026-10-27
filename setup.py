from setuptools import setup, find_packages

setup(
    name='lito',
    version='1.2.6',
    description=('An AI built entirely from scratch - no pretrained models. '
                 'Chats via its own numpy GGUF engine.'),
    author='Lito',
    packages=find_packages(),
    py_modules=['server', 'train_big', 'export_gguf'],
    include_package_data=True,
    package_data={'': ['extensions/*.py']},
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
    ],
    entry_points={
        'console_scripts': [
            'lito-server=server:main',
            'lito-train=train_big:main',
            'lito-export-gguf=export_gguf:main',
        ],
    },
)
