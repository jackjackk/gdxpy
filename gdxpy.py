import subprocess
import re
import StringIO
import pandas as pd
import sys
import time
import os
import traceback
import numpy as np

def get_gams_root():
    try:
        import enviropy
        gamsDir = (enviropy.getEnvironmentVariable('GAMSDIR').split(';'))[0]
    except:
        gamsRoot = r'C:\GAMS'
        try:
            winver = os.walk(gamsRoot).next()[1][-1]
        except:
            raise Exception('Unable to find GAMS dir')
        gamsRoot = os.path.join(gamsRoot, winver)
        gamsver = os.walk(gamsRoot).next()[1][-1]
        gamsDir =  os.path.join(gamsRoot, gamsver)
    return gamsDir

def install_gams_binding():
    gamsDir = get_gams_root()
    cmdline = 'cmd /c cd "%s" && python setup.py install' % os.path.join(gamsDir,'apifiles','Python','api')
    print cmdline
    print subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False).communicate()[0]
    
    
def print_traceback(e):
    traceback_template = '''Traceback (most recent call last):
File "%(filename)s", line %(lineno)s, in %(name)s
%(type)s: %(message)s\n'''
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback_details = {
        'filename': exc_traceback.tb_frame.f_code.co_filename,
        'lineno'  : exc_traceback.tb_lineno,
        'name'    : exc_traceback.tb_frame.f_code.co_name,
        'type'    : exc_type.__name__,
        'message' : exc_value.message, # or see traceback._some_str()
    }
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
                p3d_dict[subind.lower()] = df.xs([subind])
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
            p4d_dict[subind.lower()] = pd.Panel(p3d_dict)
        ret = pd.Panel4D(p4d_dict)
    else:
        ret = df
    return ret.fillna(0)

class gdxsymb:
    """
    Represents a GDX symbol.
    """
    def __init__(self, name, gdxsource, dim=None, stype=None, sdesc=None):
        self.name = name
        if not isinstance(gdxsource,gdxfile):
            gdxsource = gdxfile(gdxsource)
        self.gdxsource = gdxsource
        self.dim = dim
        self.stype = stype
        self.desc = sdesc

    def __repr__(self):
        return '({0}) {1}'.format(self.stype,self.desc)

    def get_values(self,reshape=True,filt=None):
        return self.gdxsource.query_symbol(self.name,reshape=reshape,filt=filt)

    def __call__(self):
        return self.get_values()


class gdxfile:
    """
    Represents a GDX file.
    """
    def __init__(self, filename=None):
        if filename == None:
           raise Exception('No GDX provided') 
        self.internal_filename = filename
        if not os.access(filename, os.R_OK):
            raise Exception('GDX "%s" not found or readable' % filename)
        cmdline = r'gdxdump.exe {0} Symbols'.format(self.internal_filename)
        symbspeclist = StringIO.StringIO(subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()[0])
        for symbspec in symbspeclist:
            m_obj = re.search(r"^\s+(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(.*)$", symbspec)
            if m_obj != None:
                sid = m_obj.group(1)
                sname = m_obj.group(2).lower()
                sdim = m_obj.group(3)
                stype = m_obj.group(4)
                sdesc = m_obj.group(5).strip()
                setattr(self, sname, gdxsymb(sname,self,sdim,stype,sdesc))

    def query_symbol(self, name, reshape=True, gamsDir=None, filt=None, fallback2csv=False):
        try:
            import gdxcc
            gdxHandle = gdxcc.new_gdxHandle_tp()
            if gamsDir == None:
                gamsDir = get_gams_root()
            gamsDir = r'C:\GAMS\win64\23.8'
            rc = gdxcc.gdxCreateD(gdxHandle, gamsDir, gdxcc.GMS_SSSIZE)
            assert rc[0],rc[1]
            assert gdxcc.gdxOpenRead(gdxHandle, self.internal_filename)[0]
            ret, symNr = gdxcc.gdxFindSymbol(gdxHandle, name)
            assert ret, "Symbol '%s' not found" % name
            ret, symName, dim, symType = gdxcc.gdxSymbolInfo(gdxHandle, symNr)
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
            assert not gdxcc.gdxClose(gdxHandle)
            assert gdxcc.gdxFree(gdxHandle)
            if symType == gdxcc.GMS_DT_SET:
                ifilt = 1
            cols = ['s%d' % x for x in range(dim)]+['val',]
            #print vtable[:5]
            df = pd.DataFrame(vtable,columns=cols)
            if ifilt != None:
                df = df.drop(df.columns[ifilt], axis=1)
            print "%d / %d records read from <%s>" % (rcounter, nrRecs, self.internal_filename)

        except Exception as e:
            if fallback2csv:
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
                raise e

        #print_traceback(e)
        #print df.columns.values
        #print df.columns.values[-1], list(df.columns.values[:-2]), df.columns.values[-2]
        #print df
        ncols = len(df.columns)
        if ncols>1:
            df = df.pivot_table(df.columns.values[-1],rows=list(df.columns.values[:-2]),cols=df.columns.values[-2])
            if reshape and (ncols>3):
                df = convert_pivottable_to_panel(df)
        else:
            df = df[df.columns.values[0]]
        self.data = df
        return df
        
def loadsymbols(slist,glist,gdxlabels=None,filt=None,reducel=False,remove_underscore=True,clear=True):
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
                      sdata_curr = gdxsymb(s,g).get_values(filt=filt)
                  except Exception as e:
                      print 'WARNING: Missing "%s" from "%s"' % (s,gid)
                      continue
                  validgdxs.append(gid)
                  nax = len(sdata_curr.axes)
                  if reducel and ('l' in sdata_curr.axes[-1]):
                        nax -= 1 
                        sdata_curr = eval("sdata_curr.ix[%s,'l']" % ','.join([':' for x in range(nax)]))
                  sdata[gid] = sdata_curr
            nvg = len(validgdxs)
            #if nvg == 1:
            #      svar = sdata[validgdxs[0]]
            if nvg >= 1:
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


