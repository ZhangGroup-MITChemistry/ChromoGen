import io
import os
from setuptools import find_packages, setup

package_dir = {
    p.replace('src','ChromoGen'):p.replace('.','/') for p in find_packages()
}
packages = list(package_dir.keys())

setup(
    name="ChromoGen",
    version="0.0.1",
    description="All code required to reproduce the results from ChromoGen: Diffusion model predicts single-cell chromatin conformations and to generate your own predictions.",
    url="https://github.com/ZhangGroup-MITChemistry/ChromoGen",
    author="Greg Schuette",
    packages=packages,
    package_dir=package_dir,
    include_package_data=True,
    package_data={
        '': ['**/*.pt']
    },
    install_requires=[
        'accelerate==0.22.0',
        'cooler==0.9.3',
        'cooltools==0.7.0',
        'einops==0.7.0',
        'h5py==3.9.0',
        'matplotlib==3.7',
        'mdtraj==1.9.9',
        'numpy==1.24.3',
        'pandas==2.2.3',
        'pyBigWig==0.3.22',
        'pycurl==7.45.2',
        'torch==2.0.1',
        'scipy==1.12.0',
        'torchaudio==2.0.2',
        'torchvision==0.15.2',
        'tqdm==4.66.1'
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/cu118'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
)

