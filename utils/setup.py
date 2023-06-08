# Thanks to https://github.com/mike-huls/toolbox
# for demonstrating to use github repo as a pip package

import setuptools

setuptools.setup(
    name = "dermis_utils",
    version = "0.0.1",
    author = "keyzaasyadda",
    url = 'https://github.com/Keyza-asyadda/DERMIS-MachineLearning.git',
    author_email = "kasyadda25@gmail.com",
    description = "Utility for bangkit capstone",
    packages = ["dermis_utils"],
    install_requires = ['tensorflow >= 2.12.0']
)