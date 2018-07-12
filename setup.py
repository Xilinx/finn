
""" """
__author__="Ken O'Brien"
__email__="kennetho@xilinx.com"


from setuptools import setup

setup(
    name = 'FINN',
    version = '0.0.dev1',
    description ='A Framework for Fast, Scalable Binarized Neural Network Inference',
    long_description='',
    url = 'https://gitenterprise.xilinx.com/XRLabs/FINN',
    author = 'Xilinx, Inc.',
    author_email = 'kennetho@xilinx.com',
    scripts = ['FINN/bin/finn'],
    test_suite = 'nose.collector',
    tests_require = ['nose'],
    license = '',
    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
    ],
    keywords = '',
    packages = ['FINN'],
    install_requires = ['']
)
