# Purpose

This module provides some convenient functions to process the data contained in a GAMS Data eXchange file (GDX). A GDX stores the values of one or more GAMS symbols, such as sets, parameters variables and equations. With this module, one can load and manipulate GDX symbols as Pandas data structures.

# Requirements

1.  Either one of the two:

    1.  Python GDX API binding installed
    
        This is the faster and preferred method. You can use `gdxpy.install_gams_binding()` from a console with enough privileges for performing the installation.
    
    2.  `gdxdump` path in the shell PATH environment variable
    
        This method is slower, and requires changing the code to use fallback2csv by default

2.  [pandas](http://pandas.pydata.org/) library (for Python data analysis) installed

    Make sure you have an up-to-date version at <http://pandas.pydata.org/>

# Usage

    # Import the module
    import gdxpy as gp
    
    # Create a GDX object
    gdxpath = r'C:\GAMS\win64\24.1\testlib_ml\trnsport.gdx'
    tgdx = gp.gdxfile(gdxpath)
    
    # Suggested way to programmatically load symbols
    a = tgdx.a()
    b = gp.gdxsymb('b', tgdx).get_values()
    
    # Suggesyed way to interactively load symbols
    gp.loadsymbols('a b', gdxpath)
    
    # You can have multiple GDX, arbitrary labels, and domain filtering
    gp.loadsymbols('big_symbol', 'long_gdx_file1 long_gdx_file_2','g1 g2',filt='interesting_element')

# DISCLAIMER

THIS SOFTWARE IS PRIVIDED "AS IS" AND COMES WITH NO WARRANTY. USE AT YOUR OWN RISK. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING BUT NOT LIMITED TO LOSS OR CORRUPTION OF DATA). USE AT YOUR OWN RISK.