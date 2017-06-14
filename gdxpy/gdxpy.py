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


RESHAPE_NONE, RESHAPE_SERIES, RESHAPE_FRAME, RESHAPE_PANEL = range(4)
RESHAPE_DEFAULT = RESHAPE_SERIES

_pyver = (sys.version_info.major, sys.version_info.minor)
_findexe = shutil.which if (_pyver[0] >= 3) else spawn.find_executable
_onwindows = (os.name == 'nt')
_gamsexe = 'gams.exe' if _onwindows else 'gams'
_gamsnotfound = 'GAMS not found: either set GAMSDIR or add "{}" path to the PATH environment variable'.format(_gamsexe)
_gamsdir = os.environ['GAMSDIR'].split(';')[-1]
_gamsexepath = os.path.join(_gamsdir, _gamsexe)
if not os.path.exists(_gamsexepath):
    _gamsexepath = _findexe(_gamsexe)
    assert _gamsexepath != None, _gamsnotfound
    _gamsdir = os.path.dirname(_gamsexepath)
_gamsbit = platform.architecture(_gamsexepath)[0][:2]
_pypath = sys.executable
_pybit = platform.architecture(_pypath)[0][:2]
assert _gamsbit == _pybit, ('GAMS bitness ({}bit) is not the same as Python bitness ({}bit).'
                                              'Please use the same'.format(_gamsbit, _pybit))

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


pd.Panel5D = pd.core.panelnd.create_nd_panel_factory(
                              klass_name   = 'Panel5D',
                              orders  = [ 'cool', 'labels','items','major_axis','minor_axis'],
                              slices  = { 'labels' : 'labels', 'items' : 'items',
                                          'major_axis' : 'major_axis', 'minor_axis' : 'minor_axis' },
                              slicer  = pd.Panel4D,
                              aliases = { 'major' : 'major_axis', 'minor' : 'minor_axis' },
                              stat_axis    = 2)


def convert_pivottable_to_panel(df):
    """
    Converts a pivot table (DataFrame) into a properly shaped Panel/Panel4D.
    """
    try:
        nl = df.index.nlevels
    except:
        return df
    if nl==2:
        p3d_dict = {}
        for subind in df.index.levels[0]:
            try:
                p3d_dict[subind] = df.xs([subind])
            except:
                pass
        ret = pd.Panel(p3d_dict)
    elif nl==3:
        p4d_dict = {}
        for subind in df.index.levels[0]:
            p3d_dict = {}
            for sub2ind in df.index.levels[1]:
                try:
                    p3d_dict[sub2ind] = df.xs([subind,sub2ind])
                except:
                    pass
            p4d_dict[subind] = pd.Panel(p3d_dict)
        ret = pd.Panel4D(p4d_dict)
    elif nl==4:
        p5d_dict = {}
        for subind in df.index.levels[0]:
            p4d_dict = {}
            for sub2ind in df.index.levels[1]:
                p3d_dict = {}
                for sub3ind in df.index.levels[2]:
                    try:
                        p3d_dict[sub3ind] = df.xs([subind,sub2ind,sub3ind])
                    except:
                        pass
                p4d_dict[sub2ind] = pd.Panel(p3d_dict)
            p5d_dict[subind] = pd.Panel4D(p4d_dict)
        ret = pd.Panel5D(p5d_dict)
    else:
        ret = df
    return ret.fillna(0)


def get_last_error(context, gdx_handle):
    return "Error in {}: {}".format(context, +gdxcc.gdxErrorStr(gdx_handle,gdxGetLastError(gdx_handle))[1])


class GdxSymb:
    """
    Represents a GDX symbol.
    """
    def _init(self, gdx, sinfo=None, name=None, dim=None, stype=None, desc=None):
        self.gdx = gdx
        self.values = None
        self.filtered = False
        if sinfo != None:
            self.name = sinfo['name']
            self.dim = sinfo['dim']
            self.stype = sinfo['stype']
            self.desc = sinfo['desc']
        if name != None:
            self.name = name
        if dim != None:
            self.dim = dim
        if stype != None:
            self.stype = stype
        if desc != None:
            self.desc = desc


    def _repr(self):
        return '({0}) {1}'.format(self.stype,self.desc)


    def get_values(self,filt=None,idval=None,reshape=RESHAPE_DEFAULT,reset=False):
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
            ret = self.gdx.query(self.name,reshape=reshape,filt=filt,idval=idval)
            self.values = ret
            self.filtered = (filt != None)

        return ret

    def _call(self,filt=None,idval=None,reset=False,reshape=RESHAPE_DEFAULT):
        return self.get_values(filt=filt,idval=idval,reset=reset,reshape=reshape)


class GdxFile:
    """
    Represents a GDX file.
    """

    def _init(self, filename=None,gamsdir=None):
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


    def _del(self):
        self.close()


    def get_sid_info(self,j):
        '''Return a dict of metadata for symbol with ID j.'''
        h = self.gdx_handle
        r, name, dims, stype = gdxcc.gdxSymbolInfo(h, j)
        assert r, '%d is not a valid symbol number' % j
        r, records, userinfo, description = gdxcc.gdxSymbolInfoX(h, j)
        assert r, '%d is not a valid symbol number' % j
        return {'name':name, 'stype':stype, 'desc':description, 'dim':dims}


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


    def query(self, name, reshape=RESHAPE_DEFAULT, filt=None, idval=None):
        '''
        Query attribute `idval` from symbol `name`, and return a data structure shaped according to `reshape`.
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
        vtable = [None]*(nrRecs)
        rcounter = 0
        rowtype = None
        if filt != None:
            if isinstance(filt,list):
                filt = '^({0})$'.format('|'.join([re.escape(x) for x in filt]))
            if isinstance(filt, str):
                filt_func = re.compile(filt, re.IGNORECASE).match
            else:
                filt_func = filt
        for i in range(nrRecs):
            vrow = [None]*(dim+1)
            ret, elements, values, afdim = gdxcc.gdxDataReadStr(gdx_handle)
            assert ret, get_last_error('gdxDataReadStr', gdx_handle)
            if (filt != None):
                match_filt = False
                for e in elements:
                    m = filt_func(e)
                    if m != None:
                        match_filt = True
                        break
                if not match_filt:
                    continue
            d = -1
            for d in range(dim):
                try:
                    vrow[d] = int(elements[d])
                except:
                    vrow[d] = elements[d].lower()
            vrow[d+1] = values[idval]
            vtable[rcounter] = vrow
            rcounter += 1
        gdxcc.gdxDataReadDone(gdx_handle)
        cols = ['s%d' % x for x in range(dim)]+['val',]
        df = pd.DataFrame(vtable[:rcounter],columns=cols)
        logger.debug("%d / %d records read from <%s>" % (rcounter, nrRecs, self.internal_filename))
        if symType == gdxcc.GMS_DT_SET:
            reshape = RESHAPE_SERIES
        df = dfreshape(df, reshape)
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


def dfreshape(df, reshape):
    ncols = len(df.columns)
    vcol = df.columns.values[-1]
    if ncols == 1:
        return df[vcol][0]
    idxcols = list(df.columns.values)[:-1]
    if (reshape > RESHAPE_SERIES) and (ncols > 2):
        df = df.pivot_table(vcol, index=idxcols[:-1],
                                columns=idxcols[-1])
        if (reshape == RESHAPE_PANEL) and (ncols > 3):
            df = convert_pivottable_to_panel(df)
    elif reshape >= RESHAPE_SERIES:
        df = df.set_index(idxcols)[vcol]
    return df


def gload(smatch, gpaths=None, glabels=None, filt=None, reducel=False,
          remove_underscore=True, clear=True, single=True, reshape=RESHAPE_DEFAULT,
          returnfirst=False, lowercase=True, lamb=None, verbose=True):
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
                    sdata_curr = gdxobjs[ig].query(s,filt=filt,reshape=reshape)
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
                elif (reshape==RESHAPE_PANEL) and (isinstance(sdata_curr, pd.DataFrame)):
                    df = pd.Panel(sdata)
                elif (reshape==RESHAPE_PANEL) and (isinstance(sdata_curr, pd.Panel)):
                    df = pd.Panel4D(sdata)
                elif (reshape == RESHAPE_PANEL) and (isinstance(sdata_curr, pd.Panel4D)):
                    df = pd.Panel5D(sdata)
                elif (reshape == RESHAPE_PANEL) and (isinstance(sdata_curr, pd.Panel5D)):
                    raise Exception('Panel6D not supported')
                else:
                    if isinstance(sdata_curr, float):
                        df = pd.Series(sdata)[validgdxs]
                    else:
                        df = pd.concat(sdata, keys=validgdxs)
                    if reshape==RESHAPE_NONE:
                        df.reset_index(inplace=True)
                        col2drop = df.columns[1]
                        df.drop(col2drop, axis=1, inplace=True)
                        ncols = len(df.columns)
                        df.columns = ['s{}'.format(x) for x in range(ncols-1)] + ['val',]
                    elif reshape>=RESHAPE_SERIES:
                        for i in range(len(df.index.levels)):
                            df.index.levels[i].name = 's{}'.format(i)
                        if reshape>=RESHAPE_FRAME:
                            try:
                                df.columns.name = 's{}'.format(i+1)
                                df = df.stack().unstack(0)
                            except:
                                df = df.unstack(0)
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
                        sold = _builtins[s]
                        if len(sold.shape) == len(svar.shape):
                            if verbose: print('Augmenting',s)
                            for c in svar.axes[0]:
                                sold[c] = svar[c]
                            svar = sold
                    except:
                        pass
                else:
                    _builtins[s] = svar


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


def loadsymbols(slist,glist,gdxlabels=None,filt=None,reducel=False,remove_underscore=True,clear=True,single=True,reshape=True,returnfirst=False):
    gload(slist,glist,gdxlabels,filt,reducel,remove_underscore,clear,single,reshape,returnfirst)

# Main initialization code
#load_gams_binding()

gload.last_gpaths = None
gload.last_glabels = None

