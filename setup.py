from distutils.core import setup

setup(name='gdxpy',
      version='0.2.0',
      description='Manipulate GAMS GDX files as Pandas data structures for convenient computation and visualization',
      url='https://github.com/jackjackk/gdxpy',
      author='Giacomo Marangoni',
      author_email='jackjackk@gmail.com',
      license='MIT',
      keywords='gdx gams pandas',
      entry_points={  # Optional
          'console_scripts': [
            'gdxpy = gdxpy.cli:main',
        ],
      },
      packages=['gdxpy',],
      install_requires=['numpy', 'pandas'],
)
