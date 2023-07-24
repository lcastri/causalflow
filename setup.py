from setuptools import setup
from fpcmci.version import VERSION

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


INSTALL_REQUIRES = ["tigramite>=5.1.0.3",
                    "pandas>=1.5.2",
                    "netgraph>=4.10.2",
                    "networkx>=2.8.6",
                    "ruptures>=1.1.7",
                    "scikit_learn>=1.1.3",
                    "torch>=1.11.0", 
                    "gpytorch>=1.4",       
                    "dcor>=0.5.3",
                    "h5py>=3.7.0"   
                    ]

setup(
    name = 'fpcmci',
    version = VERSION,    
    description = 'A causal discovery Python package',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/lcastri/fpcmci',
    author = 'Luca Castri',
    author_email = 'lucacastri94@gmail.com',
    packages = ['fpcmci', "fpcmci.preprocessing", "fpcmci.preprocessing.subsampling_methods", "fpcmci.basics", "fpcmci.selection_methods", "fpcmci.graph"],
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
