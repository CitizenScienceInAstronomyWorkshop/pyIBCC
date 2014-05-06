from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext

# From http://stackoverflow.com/a/21621689 because pip is total garbage
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

setup(
        name='pyIBCC',
        version='0.1.0',
        author='Edwin Simpson',
        author_email='',
        packages=['pyIBCC'],
        scripts=['bin/pyibcc'],
        url='https://github.com/CitizenScienceInAstronomyWorkshop/pyIBCC',
        license='LICENSE',
        description='Python implementation of IBCC',
        long_description=open('README.md').read(),
        cmdclass={'build_ext':build_ext},
        setup_requires=['numpy'],
        dependency_links=[
            'https://github.com/scikit-learn/scikit-learn/archive/1928f6f01ec93c14de1ef127824951a68cfda1a3.tar.gz#egg=scikit-learn-0.15-git'
            ],
        install_requires=[
            'scikit-learn==0.15-git',
            'scipy',
            'numpy'
            ]
        )
