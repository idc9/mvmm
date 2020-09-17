from setuptools import setup, find_packages

# def readme():
#     with open('README.rst') as f:
#            return f.read()

# time, abc, numbers, copy, textwrap
# os, json, argparase, re

# TODO: don't specify these for the MVMM package here
install_requires = ['numpy',
                    'pandas',
                    'scipy',
                    'scikit-learn',
                    'seaborn',
                    'matplotlib',
                    'tqdm',
                    'joblib',
                    'networkx',
                    'statsmodels',
                    'cvxpy'
                    ]


setup(name='mvmm',
      version='0.0.2',
      description='Multi-view mixture modeling.',
      author='Iain Carmichael',
      author_email='idc9@cornell.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
