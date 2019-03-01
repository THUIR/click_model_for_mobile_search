from distutils.core import setup
import glob

from setuptools import setup

def read_md(file_name):
    try:
        from pypandoc import convert
        return convert(file_name, 'rest')
    except:
        return ''

setup(
    name='clickmodels',
    version='2.0.0',
    author='Jiaxin Mao',
    packages=['clickmodels'],
    scripts=glob.glob('bin/*.py'),
    url='https://github.com/defaultstr/clickmodels',
    license='LICENSE',
    description='Click models for mobile search, forked from the project by Aleksandr Chuklin (https://github.com/varepsilon/clickmodels)',
    long_description=read_md('README.md'),
    install_requires=[],
)
