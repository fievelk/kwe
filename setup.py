from setuptools import setup, find_packages

setup(name='kwe',
      version='0.1',
      description='KWE - KeyWord Extraction in Python.',
      url='https://github.com/fievelk/kwe',
      author='Pierpaolo Pantone',
      author_email='pierpaolo.pantone@gmail.com',
      license='MIT',
      packages=find_packages(),
      package_data={'kwe': ['data/*.txt']},
      install_requires=[
          'nltk',
      ],
      zip_safe=False)
