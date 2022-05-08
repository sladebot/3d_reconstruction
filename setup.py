from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()

setup(
    name='metacast',
    version='0.0.1',
    author='Souranil Sen',
    author_email='souranil@gmail.com',
    license='Apache 2.0',
    description='CLI Tool to do 3D reconstruction',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='',
    packages=find_packages(),
    install_requires=[requirements],
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    entry_points='''
        [console_scripts]
        convert=src.main:cli
    '''
)