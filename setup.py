import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='REL',
     version='0.1',
     # scripts=['dokr'] ,
     author="Johannes Michael",
     author_email="mick.vanhulst@gmail.com",
     description="Entity Linking package",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/mickvanhulst/rel",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
    install_requires=['tiny-tokenizer', 'flair', 'unidecode', 'segtok', 'pillow', 'torch'],
    python_requires='>=3.6',
)