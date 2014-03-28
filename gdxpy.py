import subprocess
import re
import StringIO
import pandas as pd
import sys
import time
import os
import traceback
import numpy as np

GDX_MODE_API, GDX_MODE_SHELL = range(2)
gdxcc = None
__gdxpy_winver__ = None
__gdxpy_gamsdir__ = None
__gdxpy_apidir__ = None
__gdxpy_gdxccdir__ = None
__gdxpy_mode__ = None

def get_winver():
    if sys.maxsize == 2147483647:
        winver = 'win32'
    else:
        winver = 'win64'
    return winver

def get_gams_root():
    try:
        gamsdir = os.environ['GAMSDIR'].split(';')[0]
        if not __gdxpy_winver__ in gamsdir:
            raise Exception('GAMSDIR environment variable does not refer to a %s GAMS installation' % __gdxpy_winver__)
        if not os.path.isdir(gamsdir):
            raise Exception('"%s" is not a valid GAMS dir')
    except:
        gamsRoot = os.path.join(r'C:\GAMS',__gdxpy_winver__)
        try:
            gamsver = os.walk(gamsRoot).next()[1][-1]
        except:
            raise Exception('Unable to find any valid %s GAMS installation' % __gdxpy_winver__)
        gamsdir =  os.path.join(gamsRoot, gamsver)
    return gamsdir

def load_gams_binding():
    global gdxcc
    global __gdxpy_winver__
    global __gdxpy_gamsdir__
    global __gdxpy_apidir__
    global __gdxpy_gdxccdir__
    global __gdxpy_mode__
    try:
        __gdxpy_winver__ = get_winver()
        __gdxpy_gamsdir__ = get_gams_root()
        __gdxpy_apidir__ =  os.path.join(__gdxpy_gamsdir__,'apifiles','Python','api')
        __gdxpy_gdxccdir__ =  os.path.join(__gdxpy_apidir__,'build',
                                           'lib.{0}-{1}.{2}'.format(__gdxpy_winver__,sys.version_info[0],sys.version_info[1]))
        if not os.path.isdir(__gdxpy_gdxccdir__):
            print 'Please compile the GDX bindings at path "%s" (e.g. by calling install_gams_binding())' % __gdxpy_gdxccdir__
            raise Exception('GDX Bindings missing')
        if __gdxpy_gdxccdir__ not in sys.path:
            sys.path.insert(0,__gdxpy_gdxccdir__)
        print 'Using gdxcc from %s' % __gdxpy_gdxccdir__
        import gdxcc as local_gdxcc
        gdxcc = local_gdxcc
        __gdxpy_mode__ = GDX_MODE_API
    except:
        print 'Module gdxcc not found: GDX shell mode will be used'
        __gdxpy_mode__ = GDX_MODE_SHELL

def install_gams_binding():
    #import distutils.sysconfig
    #pkgDir = distutils.sysconfig.get_python_lib()
    #cmdline = 'cd {0} && if exist build (rmdir /S build) else (echo .) && python gdxsetup.py build --compiler=mingw32 && cd build\lib.w* && copy *.* {1} && if exist {2} (del {2}) else (echo .)'.format(os.path.join(gamsdir,'apifiles','Python','api'),
    #                                                                                                                                      pkgDir, os.path.join(pkgDir,'gdxcc.pyc'))
    cmdline = '{0} && cd {1} && python gdxsetup.py clean --all && python gdxsetup.py build --compiler=mingw32'.format(__gdxpy_apidir__[:2],__gdxpy_apidir__)
    print 'A shell will be opened and the following command will be executed:'
    print cmdline
                                     
    os.system('start cmd /k "echo Close all ipython instances && pause && %s && pause && exit' % cmdline)

def print_traceback(e):
    traceback.print_exc()
    return
    traceback_template = '''Traceback (most recent call last):
File "%(filename)s", line %(lineno)s, in %(name)s
%(type)s: %(message)s\n'''
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback_details = {
        'filename': os.path.split(exc_traceback.tb_frame.f_code.co_filename)[1],
        'lineno'  : exc_traceback.tb_lineno,
        'name'    : exc_traceback.tb_frame.f_code.co_name,
        'type'    : exc_type.__name__,
        'message' : exc_value.message, # or see traceback._some_str()
    }
    print "%(type)s : %(message)s [%(filename)s.%(lineno)d]" % traceback_details
    return
    print traceback_template % traceback_details
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)
    top = traceback.extract_stack()[-1]
    print ', '.join([type(e).__name__, os.path.basename(top[0]), str(top[1])])

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
    if len(df.index.levels)==2:
        p3d_dict = {}
        for subind in df.index.levels[0]:
            try:
                p3d_dict[subind] = df.xs([subind])
            except:
                pass
        ret = pd.Panel(p3d_dict)
    elif len(df.index.levels)==3:
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
    else:
        ret = df
    return ret.fillna(0)

class gdxsymb:
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

    def get_values(self,reshape=True,filt=None):
        try:
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
            ret = self.gdx.query(self.name,reshape=reshape,filt=filt)
            self.values = ret
            self.filtered = (filt != None)

        return ret

    def __call__(self,filt=None):
        return self.get_values(filt=filt)

class gdxfile:
    """
    Represents a GDX file.
    """

    def __init__(self, filename=None,gamsdir=None):
        global __gdxpy_mode__
        # Check filename
        if filename == None:
           raise Exception('No GDX provided') 
        self.internal_filename = filename
        if not os.access(filename, os.R_OK):
            raise Exception('GDX "%s" not found or readable' % filename)
        # Identify access mode (through gdxcc API or shell)
        if __gdxpy_mode__ == GDX_MODE_API:
            try:
                self.gdxHandle = gdxcc.new_gdxHandle_tp()
                rc = gdxcc.gdxCreateD(self.gdxHandle, __gdxpy_gamsdir__, gdxcc.GMS_SSSIZE)
                assert rc[0],rc[1]
                assert gdxcc.gdxOpenRead(self.gdxHandle, self.internal_filename)[0]
            except Exception as e:
                print_traceback(e)
                print "GDX API NOT WORKING: FALLING BACK TO GDX SHELL MODE"
                __gdxpy_mode__ = GDX_MODE_SHELL
        # Access symbols as class members
        #for symb in self.get_symbols_list():
        for symb in self.get_symbols_list():
            setattr(self, symb.name.lower(), symb)
        #self.symbols = self.get_symbols_list()

    def close(self):
        h = self.gdxHandle
        gdxcc.gdxClose(h)
        gdxcc.gdxFree(h)
        
    def __del__(self):
        self.close()

    def get_sid_info(self,j):
        if __gdxpy_mode__ != GDX_MODE_API:
            raise Exception('Function "get_sid_info" not available outside GDX API mode')
        h = self.gdxHandle
        r, name, dims, stype = gdxcc.gdxSymbolInfo(h, j)
        assert r, '%d is not a valid symbol number' % j
        r, records, userinfo, description = gdxcc.gdxSymbolInfoX(h, j)
        assert r, '%d is not a valid symbol number' % j
        return {'name':name, 'stype':stype, 'desc':'', 'dim':dims}

    def get_symbols_list(self):
        slist = []
        if __gdxpy_mode__ == GDX_MODE_API:
            rc, nSymb, nElem = gdxcc.gdxSystemInfo(self.gdxHandle)
            assert rc, 'Unable to retrieve "%s" info' % self.filename
            self.number_symbols = nSymb
            self.number_elements = nElem
            slist = [None]*(nSymb+1)
            for j in range(0,nSymb+1):
                sinfo = self.get_sid_info(j)
                if j==0:
                    sinfo['name'] = 'u'
                slist[j] = gdxsymb(self,sinfo)
        elif __gdxpy_mode__ == GDX_MODE_SHELL:
            cmdline = r'gdxdump.exe {0} Symbols'.format(self.internal_filename)
            symbspeclist = StringIO.StringIO(subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()[0])
            for symbspec in symbspeclist:
                m_obj = re.search(r"^\s+(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(.*)$", symbspec)
                if m_obj != None:
                    sid = m_obj.group(1)
                    sinfo = {
                        'name': m_obj.group(2),
                        'dim': m_obj.group(3),
                        'stype': m_obj.group(4),
                        'desc' : m_obj.group(5).strip() }
                    slist.append(gdxsymb(self,sinfo))
        else:
            raise Exception('Function "get_symbols_list" not available outside GDX API/SHELL mode')
        return slist
        
    def has_symbol(self,name):
        if __gdxpy_mode__ != GDX_MODE_API:
            raise Exception('Function "get_sid_info" not available outside GDX API mode')
        ret, symNr = gdxcc.gdxFindSymbol(self.gdxHandle, name)
        return ret

    def query(self, name, reshape=True, gamsdir=None, filt=None):
        if __gdxpy_mode__ == GDX_MODE_API:
            gdxHandle = self.gdxHandle
            ret, symNr = gdxcc.gdxFindSymbol(gdxHandle, name)
            assert ret, "Symbol '%s' not found in GDX '%s'" % (name,self.internal_filename)
            sinfo = self.get_sid_info(symNr)
            dim = sinfo['dim']
            symType = sinfo['stype']
            ret, nrRecs = gdxcc.gdxDataReadStrStart(gdxHandle, symNr)
            assert ret, "Error in gdxDataReadStrStart: "+gdxcc.gdxErrorStr(gdxHandle,gdxGetLastError(gdxHandle))[1]
            
            ifilt = None
            vtable = []
            rcounter = 0
            rowtype = None
            for i in range(nrRecs):
                vrow = [None]*(dim+1)
                ret, elements, values, afdim = gdxcc.gdxDataReadStr(gdxHandle)
                assert ret, "Error in gdxDataReadStr: "+gdxcc.gdxErrorStr(gdxHandle,gdxGetLastError(gdxHandle))[1]
                if (filt != None):
                    try:
                        ifilt = elements.index(filt)
                    except:
                        continue
                for d in range(dim):
                    try:
                        vrow[d] = int(elements[d])
                    except:
                        vrow[d] = elements[d]
                vrow[d+1] = values[gdxcc.GMS_VAL_LEVEL]
                vtable.append(vrow)
                rcounter += 1
            gdxcc.gdxDataReadDone(gdxHandle)
            if symType == gdxcc.GMS_DT_SET:
                ifilt = 1
            cols = ['s%d' % x for x in range(dim)]+['val',]
            #print vtable[:5]
            #print cols
            df = pd.DataFrame(vtable,columns=cols)
            #print df
            if ifilt != None:
                df = df.drop(df.columns[ifilt], axis=1)
            #print "%d / %d records read from <%s>" % (rcounter, nrRecs, self.internal_filename)

        elif __gdxpy_mode__ == GDX_MODE_SHELL:
            cmdline = r'gdxdump.exe {0} symb={1} Format=csv NoHeader'.format(self.internal_filename, name)
            p = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
            # TODO check for return code
            # p = subprocess.Popen(cmdline +' | tr "[:upper:]" "[:lower:]"', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            strdata = p.communicate()[0] #.replace("'.","',").replace(', \r\n','\r\n').replace(" ",",")
            sepchar = ','
            strdata = strdata.replace('eps','1e-16')
            csvfile = StringIO.StringIO(strdata)
            #print strdata[:500]
            df = pd.read_csv(csvfile,sep=sepchar,quotechar='"',prefix='s',header=None,error_bad_lines=False).dropna()
        else:
            raise Exception('Function "get_symbols_list" not available outside GDX API/SHELL mode')

        #raise e,  None, sys.exc_info()[2]
        #print_traceback(e)
        #print df.columns.values
        #print df.columns.values[-1], list(df.columns.values[:-2]), df.columns.values[-2]
        #print df
        ncols = len(df.columns)
        if ncols>1:
            df = df.pivot_table(df.columns.values[-1],rows=list(df.columns.values[:-2]),cols=df.columns.values[-2])
            #print df
            if reshape and (ncols>3):
                df = convert_pivottable_to_panel(df)
        else:
            df = df[df.columns.values[0]]
        self.data = df
        return df
        
def loadsymbols(slist,glist,gdxlabels=None,filt=None,reducel=False,remove_underscore=True,clear=True,single=True):
      """
      Loads into global namespace the symbols listed in {slist}
      from the GDX listed in {glist}.
      If {reducel}==True, filter the dataset on 'l' entries only.
      If {remove_underscore}==True, symbols are loaded into the global
      namespace with their names without underscores.
      """
      if isinstance(slist,str):
            slist = slist.split(" ")
      if isinstance(glist,str):
            glist = glist.split(" ")
      if isinstance(gdxlabels,str):
            gdxlabels = gdxlabels.split(" ")
      ng = len(glist)
      nax = 0
      #print 'From GDX:\n%s' % ('\n'.join(['g%d) %s' % (ig+1,g) for ig,g in enumerate(glist)]))
      for s in slist:
            print '\n<<< %s >>>' % s
            sdata = {}
            svar = None
            validgdxs = []
            for ig,g in enumerate(glist):
                  fname, fext = os.path.splitext(g)
                  if gdxlabels == None:
                      gid = fname
                  else:
                      if isinstance(gdxlabels,int):
                          gid = 'g%d' % (ig+gdxlabels)
                      else:
                          gid = gdxlabels[ig]
                  try:
                      sdata_curr = gdxfile(g).query(s,filt=filt)
                  except Exception as e:
                      traceback.print_exc()
                      #print_traceback(e)
                      #print 'WARNING: Missing "%s" from "%s"' % (s,gid)
                      continue
                  validgdxs.append(gid)
                  nax = len(sdata_curr.axes)
                  if reducel and ('l' in sdata_curr.axes[-1]):
                        nax -= 1 
                        sdata_curr = eval("sdata_curr.ix[%s,'l']" % ','.join([':' for x in range(nax)]))
                  sdata[gid] = sdata_curr
            nvg = len(validgdxs)
            if (nvg == 1) and single==True:
                  svar = sdata[validgdxs[0]]
            elif nvg >= 1:
                  if nax == 1:
                        svar = pd.DataFrame(sdata)
                  elif nax == 2:
                        svar = pd.Panel(sdata)
                  elif nax == 3:
                        svar = pd.Panel4D(sdata)
                  elif nax == 4:
                        svar = pd.Panel5D(sdata)
                  else:
                        raise Exception('Dimension not supported')
            else:
                continue
            if remove_underscore:
                  s = s.replace('_','')
            if not clear:
                try:
                    sold = sys.modules['__builtin__'].__dict__[s]
                    if len(sold.shape) == len(svar.shape):
                        print 'Augmenting',s
                        for c in svar.axes[0]:
                            sold[c] = svar[c]
                        svar = sold
                except:
                    pass
            sys.modules['__builtin__'].__dict__[s] = svar

            
            if isinstance(svar,pd.DataFrame):
                print svar.describe()
            else:
                print svar
            time.sleep(0.01)


# Main initialization code
load_gams_binding()
