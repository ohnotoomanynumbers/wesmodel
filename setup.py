import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="wesmodel",
    version="0.0.1",
    author="Andrew MacMaster",
    author_email="a.w.macmaster@gmail.com",
    packages=["wesmodel"],
    description="A package that uses openai to perform topic modeling on a natural language dataset",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://gitlab.onefiserv.net/f9ruef2/wesmodel",
    license="MIT",
    python_requires=">=3.10.11",
    install_requires=[
        "openai>=1.12.0",
        "pandas>=2.2.0",
        "tiktoken>=0.6.0",
        "tqdm>=4.66.2"
        ]
)