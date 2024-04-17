import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="progres",
    version="0.2.1",
    author="Joe G Greener",
    author_email="jgreener@mrc-lmb.cam.ac.uk",
    description="Fast protein structure searching using structure graph embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/greener-group/progres",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license="MIT",
    keywords="protein structure search graph embedding",
    scripts=["bin/progres"],
    install_requires=["biopython", "mmtf-python", "einops"],
    include_package_data=True,
)
