from setuptools import setup, find_packages

setup(
    name='conversion_block',
    version='0.1.0',

    packages=find_packages(),
    python_requires='>=3.8',
    entry_points={
        'console_scripts': ['conversion_block=conversion_block.main:cli']
    },
    include_package_data=True,
)
