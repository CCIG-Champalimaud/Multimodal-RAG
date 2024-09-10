from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='Multimodal-RAG',
    version='0.0.1',
    description='Multimodal RAG system for medical reports',
    long_description=readme,
    author='Nuno Rodrigues',
    author_email='nuno.mvrodrigues1@gmail.com',
    #url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')) #??
)