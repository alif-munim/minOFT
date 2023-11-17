from setuptools import setup, find_packages

setup(
    name='minOFT', 
    version='0.1.0',  
    author='Alif Munim', 
    author_email='alif.munim@torontomu.ca',  
    description='A minimal implementation of orthogonal fine-tuning',  
    long_description=open('README.md').read(), 
    long_description_content_type='text/markdown',  
    url='https://github.com/alif-munim/minOFT', 
    packages=["minoft"],  
    install_requires=[
        'numpy',
        'torch',
        'Pillow', 
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache License 2.0',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6', 
    include_package_data=True, 
)