import re
import pandas as pd
import sys
import os
import traceback
import numpy as np
import glob
import itertools
import pdb
import shutil
import platform
from distutils import spawn
import builtins
import inspect
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('gdxpy')
logger.debug('Logger created')

_version = '0.2.0'
_pyver = (sys.version_info.major, sys.version_info.minor)
_findexe = shutil.which if (_pyver[0] >= 3) else spawn.find_executable
_onwindows = (os.name == 'nt')
_gamsexe = 'gams.exe' if _onwindows else 'gams'
_gamsnotfound = 'GAMS not found: either set GAMSDIR or add "{}" path to the PATH environment variable'.format(_gamsexe)
try:
    _gamsfound = False
    for _gamsdir in os.environ['GAMSDIR'].split(';'):
        _gamsexepath = os.path.join(_gamsdir, _gamsexe)
        if os.path.exists(_gamsexepath):
            _gamsfound = True
            break
    assert _gamsfound
except:
    _gamsexepath = os.path.realpath(_findexe(_gamsexe))
    assert _gamsexepath != None, _gamsnotfound
    _gamsdir = os.path.dirname(_gamsexepath)
_gamsbit = platform.architecture(_gamsexepath)[0][:2]
_pypath = sys.executable
_pybit = platform.architecture(_pypath)[0][:2]
assert _gamsbit == _pybit, ('GAMS bitness ({}bit) is not the same as Python bitness ({}bit).'
                                              'Please use the same'.format(_gamsbit, _pybit))

logger.info(f'Using gdxpy v.{_version} ({os.path.dirname(__file__)})')
logger.info('Using {} as GAMS directory'.format(_gamsdir))
try:
    import gdxcc
    _gdxccdir = os.path.dirname(inspect.getfile(gdxcc))
except:
    _gdxccdir = os.path.join(_gamsdir, 'apifiles', 'Python', 'api_{}{}'.format(*_pyver).replace('_27',''))
    if not os.path.exists(_gdxccdir):
        if _onwindows:
            bit2lab = { '32': 'win32', '64': 'win-amd64'}
            spec2builddir = lambda b, v : 'lib.{}-{}.{}'.format(bit2lab[b], v[0], v[1])
            _gdxccdir = os.path.join(_gdxccdir, 'build', spec2builddir(_pybit, _pyver))
            if not os.path.exists(_gdxccdir):
                cmdline = ('{0} && cd {1} && {2} gdxsetup.py clean --all && {2} gdxsetup.py build --compiler=mingw32'
                           .format(_gdxccdir[:2], _gdxccdir, _pypath))
                print(cmdline)
                os.system('start cmd /k "echo Close all python instances && pause && %s && pause && exit' % cmdline)
                assert False, ('Follow instructions to install Python-GDX interface for unsupported Python version.'
                               'A compiler like mingw32 required.')
        else:
            assert False, 'Please install GAMS-Python bindings on your system.'
    sys.path.insert(0, _gdxccdir)
    import gdxcc

logger.info('Using gdxcc from {}'.format(_gdxccdir))

L, M, LO, UP = (gdxcc.GMS_VAL_LEVEL,
                gdxcc.GMS_VAL_MARGINAL,
                gdxcc.GMS_VAL_LOWER,
                gdxcc.GMS_VAL_UPPER)


def get_gams_root():
    '''Get GAMS root directory path.'''
    return _gamsdir


def print_traceback(e):
    '''Print trace back for Exception e.'''
    traceback.print_exc()
    return



def get_last_error(context, gdx_handle):
    return "Error in {}: {}".format(context, +gdxcc.gdxErrorStr(gdx_handle,gdxGetLastError(gdx_handle))[1])


class GdxSymb:
    """
    Represents a GDX symbol.
    """
    def __init__(self, gdx, sinfo=None, name=None, dim=None, stype=None, desc=None):
        self.gdx = gdx
        self.values = None
        self.filtered = False
        if sinfo != None:
            self.name = sinfo['name']
            self.dim = sinfo['dim']
            self.stype = sinfo['stype']
            self.desc = sinfo['desc']
            self.dom = sinfo['dom']
        if name != None:
            self.name = name
        if dim != None:
            self.dim = dim
        if stype != None:
            self.stype = stype
        if desc != None:
            self.desc = desc


    def __repr__(self):
        return '({0}) {1}'.format(self.stype,self.desc)


    def get_values(self,filt=None,idval=None,reset=False):
        try:
            if reset:
                raise Exception('forced reload')
            axs = self.values.axes
            assert not self.filtered, 'If was loaded filtered before, need for reload from source'
            ret = self.values
            if filt != None:
                bfound = False
                for iax, ax in enumerate(axs):
                    for x in ax:
                        if filt == x:
                            bfound = True
                            break
                if not bfound:
                    raise Exception('Element "%s" not found' % str(filt))
                ret = self.values.xs(filt,axis=iax)
        except:
            ret = self.gdx.query(self.name,filt=filt,idval=idval)
            self.values = ret
            self.filtered = (filt != None)

        return ret

    def __call__(self,filt=None,idval=None,reset=False):
        return self.get_values(filt=filt,idval=idval,reset=reset)


class GdxFile():
    """
    Represents a GDX file.
    """

    def __init__(self, filename=None,gamsdir=None):
        assert os.access(filename, os.R_OK) != None, 'Gdx file "{}" not found or readable!'.format(filename)
        self.internal_filename = os.path.abspath(filename)
        self.gdx_handle = gdxcc.new_gdxHandle_tp()
        rc = gdxcc.gdxCreateD(self.gdx_handle, _gamsdir, gdxcc.GMS_SSSIZE)
        assert rc[0], rc[1]
        assert gdxcc.gdxOpenRead(self.gdx_handle, self.internal_filename)[0], 'Unable to read "{}"!'.format(filename)
        for symb in self.get_symbols_list():
            setattr(self, symb.name.lower(), symb)


    def close(self):
        '''Close Gdx file and free up resources.'''
        h = self.gdx_handle
        gdxcc.gdxClose(h)
        gdxcc.gdxFree(h)


    def __del__(self):
        self.close()


    def get_sid_info(self,j):
        '''Return a dict of metadata for symbol with ID j.'''
        h = self.gdx_handle
        r, name, dims, stype = gdxcc.gdxSymbolInfo(h, j)
        assert r, '%d is not a valid symbol number' % j
        r, records, userinfo, description = gdxcc.gdxSymbolInfoX(h, j)
        assert r, '%d is not a valid symbol number' % j
        r, domains = gdxcc.gdxSymbolGetDomainX(h, j)
        return {'name':name, 'stype':stype, 'desc':description, 'dim':dims, 'dom': domains}


    def get_symbols_list(self):
        '''Return a list of GdxSymb found in the GdxFile.'''
        slist = []
        rc, nSymb, nElem = gdxcc.gdxSystemInfo(self.gdx_handle)
        assert rc, 'Unable to retrieve "%s" info' % self.filename
        self.number_symbols = nSymb
        self.number_elements = nElem
        slist = [None]*(nSymb+1)
        for j in range(0,nSymb+1):
            sinfo = self.get_sid_info(j)
            if j==0:
                sinfo['name'] = 'universal_set'
            slist[j] = GdxSymb(self,sinfo)
        return slist


    def has_symbol(self,name):
        ret, symNr = gdxcc.gdxFindSymbol(self.gdx_handle, name)
        return ret


    def query(self, name, filt=None, idval=None, idxlower=True):
        '''
        Query attribute `idval` from symbol `name`, and return a Pandas data structure.
        '''
        gdx_handle = self.gdx_handle
        ret, symNr = gdxcc.gdxFindSymbol(gdx_handle, name)
        assert ret, "Symbol '%s' not found in GDX '%s'" % (name, self.internal_filename)
        sinfo = self.get_sid_info(symNr)
        dim = sinfo['dim']
        symType = sinfo['stype']
        ret, nrRecs = gdxcc.gdxDataReadStrStart(gdx_handle, symNr)
        assert ret, get_last_error('gdxDataReadStrStart', gdx_handle)
        if idval is None:
            if symType == gdxcc.GMS_DT_EQU:
                idval = gdxcc.GMS_VAL_MARGINAL
            else:
                idval = gdxcc.GMS_VAL_LEVEL
        ifilt = None
        istar = 1
        dom = sinfo['dom']
        for idom, adom in enumerate(dom):
            if adom == '*':
                dom[idom] = f's{istar}'
                istar += 1
        dom.append(name)
        vtable = np.zeros(nrRecs, dtype = list(zip(dom, [object]*dim + ['f8',])))
        rcounter = 0
        rowtype = None
        if filt != None:
            if ',' in filt:
                
                relist = [re.compile(f'^({x})$', re.IGNORECASE).match if x != '' else (lambda a: True) for x in filt.split(',')]
                filt_func = lambda elems: np.all([m(e) is not None for m,e in zip(relist,elems)])
            else:
                if isinstance(filt,list):
                    filt = '^({0})$'.format('|'.join([re.escape(x) for x in filt]))
                if isinstance(filt, str):
                    filt_func = re.compile(filt, re.IGNORECASE).match
                else:
                    filt_func = filt
        for i in range(nrRecs):
            ret, elements, values, afdim = gdxcc.gdxDataReadStr(gdx_handle)
            assert ret, get_last_error('gdxDataReadStr', gdx_handle)
            if (filt != None):
                if not any(map(filt_func, elements)):
                    continue
            d = -1
            for idom in range(dim):
                try:
                    vtable[rcounter][idom] = int(elements[idom])
                except:
                    vtable[rcounter][idom] = elements[idom].lower() if idxlower else elements[idom]
            vtable[rcounter][idom+1] = values[idval]
            rcounter += 1
        gdxcc.gdxDataReadDone(gdx_handle)
        df = pd.DataFrame(vtable[:rcounter]).set_index(dom[:-1])  #,columns=cols)
        name_specs = []
        if dim > 1:
            levs2drop = []
            for ilev, lev in enumerate(df.index.levels):
                if len(lev) == 1:
                    levs2drop.append(ilev)
                    name_specs.append(lev[0])
            df = df.droplevel(levs2drop, axis=0)
            if len(name_specs) > 0:
                df.set_axis([f'{name} [{",".join(name_specs)}]',], axis=1, inplace=True)
        df = df.iloc[:,0]
        logger.debug("%d / %d records read from <%s>" % (rcounter, nrRecs, self.internal_filename))
        if symType == gdxcc.GMS_DT_SET:
            df = df.index
        self.data = df
        return df


def expandmatch(m):
    m = m.replace(' ',';').split(';')
    if len(m)>1:
        m = '({0})'.format('|'.join(m))
    else:
        m = m[0]
    if m[-1] != '$':
        m += '$'
    return m

def gslice(ds,kspec,op=None):
    if isinstance(kspec,str):
        kspec = kspec.split(';')
    for k in kspec:
        vfound = []
        for ia,a in enumerate(ds.axes):
            if isinstance(k,str) and ':' in k:
                rlim = k.split(':')
                k = range(int(rlim[0]),int(rlim[1])+1)
            if isinstance(k,str):
                for v in a.values:
                    if not isinstance(v,str):
                        break
                    if re.match(k,v,re.I):
                        vfound.append(v)
            elif isinstance(k,list):
                if np.in1d(k,a.values).all():
                    vfound = k
            #print vfound, a.values
            nv = len(vfound)
            if nv>0:
                if nv>1:
                    if ia>0:
                        filterfmt = '[%s,{0}]'%(','.join([':']*ia))
                    else:
                        filterfmt = '[{0}]'
                    ds = eval('ds.ix'+filterfmt.format(str(vfound)))
                    if op=='sum':
                        ds = ds.sum(axis=ia)
                else:
                    ds = ds.xs(vfound[0],axis=ia)
                break
    return ds


def gsum(ds,kspec):
    return gslice(ds,kspec,'sum')


def expandlist(l,auxl=None):
    if (l is None) or isinstance(l,int):
        return l
    if hasattr(l, '_call'):
        return [l(x) for x in auxl]
    if isinstance(l,str):
        l = l.replace(' ',';').split(';')
    ret = []
    for i in l:
        curlynext = i.find('{')
        curlygroups = []
        curlyend = -1
        while curlynext > -1:
            # append a group with substr up to the open curly brace
            curlygroups.append([i[curlyend+1:curlynext]])
            # find the closing curly bracket
            curlyend = i.find('}',curlynext)
            curlyliststr = i[curlynext+1:curlyend]
            doublepoint = curlyliststr.find('..')
            if doublepoint>-1:
                curlylist = ['%d'%x for x in range(int(curlyliststr[:doublepoint]),int(curlyliststr[doublepoint+2:])+1)]
            else:
                curlylist = curlyliststr.split(',')
            curlygroups.append(curlylist)
            curlynext = i.find('{',curlynext+1)
        curlygroups.append([i[curlyend+1:]])
        for combo in itertools.product(*curlygroups):
            j = ''.join(combo)
            if ('*' in j) or ('?' in j):
                ret.extend(glob.glob(j))
            else:
                ret.append(j)
    return ret


def gload(smatch, gpaths=None, glabels=None, filt=None, reducel=False,
          remove_underscore=True, clear=True, single=True,
          idxlower=True, returnfirst=False, lowercase=True, lamb=None, verbose=True,
          idval=None):
      """
      Loads into global namespace the symbols listed in {slist}
      from the GDX listed in {gpaths}.
      If {reducel}==True, filter the dataset on 'l' entries only.
      If {remove_underscore}==True, symbols are loaded into the global
      namespace with their names without underscores.
      """
      # Normalize the match string for symbols
      if smatch[0] == '@':
          returnfirst = True
          smatch = smatch[1:]
      smatch = expandmatch(smatch)
      # Build gdxobj list and
      if isinstance(gpaths,list) and isinstance(gpaths[0],GdxFile):
          gpaths = [g.internal_filename for g in gpaths]
          gdxobjs = gpaths
      elif not gpaths is None:
          gpaths = expandlist(gpaths)
          gdxobjs = [GdxFile(g) for g in gpaths]
      else:
          gpaths = gload.last_gpaths
          gdxobjs = [GdxFile(g) for g in gpaths]
          glabels = gload.last_glabels
      # Normalize the list of labels for gdxs
      gload.last_gpaths = gpaths
      gload.last_glabels = glabels
      glabels = expandlist(glabels,gpaths)
      all_symbols = set()
      for g in gdxobjs:
          all_symbols |= set([x.name for x in g.get_symbols_list()])
      ng = len(gpaths)
      nax = 0
      if verbose: print(smatch)
      svar2ret = []
      for s in all_symbols:
            m = re.match(smatch,s, re.M|re.I)
            if not m:
                continue
            if verbose: print('\n<<< %s >>>' % s)
            sdata = {}
            svar = None
            validgdxs = []
            for ig,g in enumerate(gpaths):
                fname, fext = os.path.splitext(g)
                if glabels == None:
                    gid = fname
                else:
                    if isinstance(glabels,int):
                        gid = 'g%d' % (ig+glabels)
                    else:
                        gid = glabels[ig]
                try:
                    sdata_curr = gdxobjs[ig].query(s,filt=filt,idval=idval,idxlower=idxlower)
                    sdata[gid] = sdata_curr
                except Exception as e:
                    #traceback.print_exc()
                    if verbose:
                        print_traceback(e)
                        print('WARNING: Missing "%s" from "%s"' % (s,gid))
                    continue
                validgdxs.append(gid)
            nvg = len(validgdxs)
            if nvg>1:
                if isinstance(sdata_curr, pd.Index):
                    df = pd.concat({gid: pd.Series(1, x) for gid, x in sdata.items()}, keys=validgdxs).index
                else:
                    if isinstance(sdata_curr, float):
                        df = pd.Series(sdata)[validgdxs]
                    else:
                        df = pd.concat(sdata, keys=validgdxs)
            else:
                df = sdata_curr
            try:
                df.name = s
            except:
                pass
            svar = df
            if remove_underscore:
                  s = s.replace('','')
            if lowercase:
                s = s.lower()
            if not lamb is None:
                svar = lamb(svar)
            if not returnfirst:
                if not clear:
                    try:
                        sold = __builtins__[s]
                        if len(sold.shape) == len(svar.shape):
                            if verbose: print('Augmenting',s)
                            for c in svar.axes[0]:
                                sold[c] = svar[c]
                            svar = sold
                    except:
                        pass
                else:
                    __builtins__[s] = svar

            if verbose:
                logprint = logger.info if verbose else logger.debug
                if isinstance(svar, pd.DataFrame):
                    logprint('Rows   : {} ... {}'.format(str(svar.index[0]), str(svar.index[-1])))
                    colwidth = np.max([len(str(svar.columns[i])) for i in range(len(svar.columns))])
                    logprint('Columns: {}'.format('\n         '.join([('{:<%d} = {} ... {}'%colwidth).format(
                        str(svar.columns[i]), svar.iloc[0,i], svar.iloc[-1,i]) for i in range(len(svar.columns))])))
                elif isinstance(svar, pd.Series):
                    logprint('Index  : {} ... {}'.format(str(svar.index[0]), str(svar.index[-1])))
                else:
                    logprint(svar)
            if returnfirst:
                svar2ret.append(svar)
      if returnfirst:
          if len(svar2ret) == 1:
              svar2ret = svar2ret[0]
          return svar2ret


def loadsymbols(slist,glist,gdxlabels=None,filt=None,reducel=False,remove_underscore=True,clear=True,single=True,returnfirst=False):
    gload(slist,glist,gdxlabels,filt,reducel,remove_underscore,clear,single,returnfirst)

# Main initialization code
#load_gams_binding()

gload.last_gpaths = None
gload.last_glabels = None

