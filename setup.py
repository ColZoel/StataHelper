from setuptools import setup, find_packages
setup(
    name='StataHelper',
    version='1.0.0',
    packages=find_packages(exclude=['tests*']),
    url='https://github.com/ColZoel/StataHelper',
    license='MIT',
    author='Collin Zoeller',
    author_email='zoellercollin@gmail.com',
    description='Simple parallelization wrapper over Pystata',
    keywords=['Stata', 'Pystata', 'parallelization', 'wrapper'],
)
