import argparse
import os
import glob
import logging
import time
import sys
import hashlib
import gdxpy.gdxpy as gp
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('gdxpy')


def main():
    parser = argparse.ArgumentParser(description='Witch launcher')
    parser.add_argument('gdxfile', nargs='+', type=str,
                        help='Input gdx file(s)')
    parser.add_argument('-s', '--symb', type=str,
                        help='Symbol match')
    parser.add_argument('-f', '--filt', type=str, default=None,
                        help='Symbol filter')
    parser.add_argument('-u', '--unstack', type=int, default=None,
                        help='Level to unstack')
    args = parser.parse_args()
    pd.set_option('display.max_colwidth', 16)
    df = gp.gload(args.symb, args.gdxfile, glabels=[f'..{g[-18:-4]}' for g in args.gdxfile], filt=args.filt, returnfirst=True, verbose=False)
    df[df>1e9] = np.nan
    nmax = df.max().max()
    log10nmax = np.log10(nmax)
    if log10nmax < -3:
        pd.set_option('display.float_format', lambda x: '%.2e' % x)
    else:
        pd.set_option('display.float_format', lambda x: (f'%.3f') % x)
    if args.unstack is not None:
        df = df.unstack(args.unstack)
    sspecs = ''
    try:
        levs2drop = []
        for i, lev in enumerate(df.index.levels):
            if len(lev) == 1:
                levs2drop.append(i)
                sspecs += f'[{lev[0]}]'
        df.index = df.index.droplevel(levs2drop)
    except:
        pass
    print(df)
    print(sspecs)
    


if __name__ == '__main__':
    main()
