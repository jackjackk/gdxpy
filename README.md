# Purpose

This module provides some convenient functions to process the data contained in a GAMS Data eXchange file (GDX). A GDX stores the values of one or more GAMS symbols, such as sets, parameters variables and equations. With this module, one can load and manipulate GDX symbols as Pandas data structures.

# Requirements

1.  `gdxdump` path in the shell PATH environment variable

2.  [pandas](http://pandas.pydata.org/) library (for Python data analysis) installed

# Usage

    import gdxpy as gp
    gdxpath = r'C:\GAMS\win64\24.1\testlib_ml\trnsport.gdx'
    tgdx = gp.gdxfile(gdxpath)
    # 3 ways of loading symbols into Pandas data structures
    a = tgdx.a()
    b = gp.gdxsymb('b', tgdx).populate()
    gp.loadsymbols('c', gdxpath)
    # Now a, b, c can be used for calculations/plotting

# DISCLAIMER

THIS SOFTWARE IS PRIVIDED "AS IS" AND COMES WITH NO WARRANTY. USE AT YOUR OWN RISK. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING BUT NOT LIMITED TO LOSS OR CORRUPTION OF DATA). USE AT YOUR OWN RISK.