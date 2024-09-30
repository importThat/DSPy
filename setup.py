from setuptools import setup

setup(
    name = "dsproc",
    version = "0.0.0a1",
    install_requires=["matplotlib >= 3.8.0", "numpy >= 1.2.0", "scipy >= 1.1.0"],
    description="dsproc: a powerful digital signals processing toolkit",
    long_description="dsproc is a Python package that enables the analysis and processing of digital radio signals using "
                "an intuitive and approachable framework. It supports end to end digital communcations and gives users "
                "the ability to encode and modulate data into radio waves of many types. "
                "Source code - https://github.com/importThat/dsproc"

)