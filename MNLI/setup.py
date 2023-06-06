from setuptools import setup, find_packages

# build long description from README.md
with open("README.md", "r") as fh:
    long_description = fh.read()

listed_requires = [
    # torch
    "torch==1.6.0", #+cu101 is broken
    "torchvision==0.7.0", #+cu101
     # torch geometric related
    "torch_geometric==1.6.0",
    "torch_scatter==2.0.5",
    "torch_sparse==0.6.7",
    # allennlp, should have a lot of dependencies intalled
    "allennlp==1.1.0rc3",
    "allennlp-models==1.1.0rc3",
    # transformers
    "transformers==3.0.2",
    # utils,typing
    "tqdm",
    "typing",
    "pathlib",
]   
    
with open('requirements.txt') as fid:
    requires = [line.strip() for line in fid]    
    
setup(
    name="syn_nli", # Replace with your own username
    version="0.0.3", #version
    author="ytlin",
    author_email="0312fs3@gmail.com",
    description="package for the paper Syntax Aware Natural Language Inference@<link>",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EazyReal/2020-IIS-internship",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="allennlp NLP deep learning machine",
    license="Apache",
    packages=find_packages(
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
            "test_fixtures",
            "test_fixtures.*",
            "benchmarks",
            "benchmarks.*",
            "previous_srcs"
        ]
    ),
    install_requires=listed_requires
    ,
    entry_points={"console_scripts": ["src=src.__main__:run"]},
    include_package_data=True,
    python_requires=">=3.6.1",
    zip_safe=False,
)