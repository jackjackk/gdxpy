


1 Purpose
---------

This module provides some convenient functions to process the data contained in a GAMS Data eXchange file (GDX). A GDX stores the values of one or more GAMS symbols, such as sets, parameters variables and equations. With this module, one can load and manipulate GDX symbols as Pandas data structures.

2 Requirements
--------------

- Python (tested with version 3.6).

- up-to-date `numpy <http://www.numpy.org/%E2%80%8E>`_ module.

- up-to-date `pandas <http://pandas.pydata.org/>`_ module.

- GAMS installation with the same architecture of Python (e.g. 32bit Python will work with 32bit GAMS).

Note that if you have a Python version not directly supported by your GAMS distribution, you will need to recompile the GDX-Python bindings. This module will warn you if this is the case and provide you with instructions on how to proceed.

3 Usage
-------

.. code:: python

    # Import modules
    import gdxpy as gp
    import os

    # Create a GDX object
    gdxpath = os.path.join(gp.get_gams_root(), 'testlib_ml', 'trnsport.gdx')
    tdata = gp.GdxFile(gdxpath)

    # Suggested way to programmatically load symbols
    a = tdata.a() # or...
    b = tdata.query('b') # or...
    c = gp.gload('@c', gdxpath)

    # Suggested way to interactively load symbols
    gp.gload('a b c', gdxpath)

    # You can have multiple GDX, arbitrary labels, and domain filtering, e.g.
    # gp.gload('big_symbol', 'long_gdx_file1 long_gdx_file_2','g1 g2',filt='interesting_element')

- By default, a Pandas Series is returned for data, a Pandas Index for sets, and a float for scalars. To change this behavior, choose an appropriate ``reshape`` value to either obtain a tidy DataFrame (``RESHAPE_NONE``), an unstacked Series (``RESHAPE_FRAME``) or a multi-dimensional Panel (``RESHAPE_PANEL``).

- Note the ``@`` in ``gload`` as a shortcut for returning data instead of loading it in the environment.

4 License
---------

The MIT License (MIT)

Copyright (c) 2014-2017 Giacomo Marangoni

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
