from setuptools import setup, find_packages

# build long description from README.md
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="syn_nli", # Replace with your own username
    version="0.0.1",
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
    install_requires=[
        "torch>=1.6.0,<1.7.0",
        "allennlp==1.1.0rc3",
        "pytorch_geometic" # change this later
    ],
    entry_points={"console_scripts": ["src=src.__main__:run"]},
    include_package_data=True,
    python_requires=">=3.6.1",
    zip_safe=False,
)