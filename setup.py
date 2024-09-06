from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    print("long_description: ", long_description)

setup(
    name='StataHelper',
    version='1.0.1',
    packages=find_packages(exclude=['tests*']),
    url='https://github.com/ColZoel/StataHelper',
    license='MIT',
    author='Collin Zoeller',
    author_email='zoellercollin@gmail.com',
    description='Simple parallelization wrapper over Pystata',
    long_description_content_type="text/markdown",
    long_description="long_description",
    keywords=['Stata', 'Pystata', 'parallelization', 'wrapper'],
)
