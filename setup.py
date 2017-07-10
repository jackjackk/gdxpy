from distutils.core import setup

setup(name='gdxpy',
      version='0.1.3',
      description='Manipulate GAMS GDX files as Pandas data structures for convenient computation and visualization',
      url='https://github.com/jackjackk/gdxpy',
      author='Giacomo Marangoni',
      author_email='jackjackk@gmail.com',
      license='MIT',
      keywords='gdx gams pandas',
      packages=['gdxpy',],
      install_requires=['numpy', 'pandas'],
)
