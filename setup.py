from setuptools import setup
from fpcmci.version import VERSION

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


INSTALL_REQUIRES = ["pandas>=1.5.2",
                    "netgraph>=4.10.2",
                    "networkx>=2.8.6",
                    "ruptures>=1.1.7",
                    "scikit_learn>=1.1.3",
                    "torch>=1.11.0",
                    "gpytorch>=1.4",
                    "dcor>=0.5.3",
                    "h5py>=3.7.0",   
                    "jpype1>=1.5.0",
                    "mpmath>=1.3.0",  
                    "causalnex>=0.12.1",
                    "lingam>=1.8.2",
                    "tigramite>=5.1.0.3",
                    "pyopencl>=2024.1",
                    ]

setup(
    name = 'causalflow',
    version = VERSION,    
    description = 'A Python library for Causal Inference in Time-Series domain',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/lcastri/causalflow',
    author = 'Luca Castri',
    author_email = 'lucacastri94@gmail.com',
    packages = ['causalflow', "causalflow.preprocessing", "causalflow.preprocessing.subsampling_methods", "causalflow.basics", "causalflow.selection_methods", "causalflow.graph", "causalflow.random_system"],
    python_requires='>=3',
    install_requires = INSTALL_REQUIRES,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',  
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)