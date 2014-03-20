import subprocess
import re
import StringIO
import pandas as pd
import sys
import time
import os
import traceback
traceback_template = '''Traceback (most recent call last):
  File "%(filename)s", line %(lineno)s, in %(name)s
%(type)s: %(message)s\n'''

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

    def get_values(self,reshape=True):
        return self.gdxsource.query_symbol(self.name, reshape)

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

    def query_symbol(self, name, reshape=True):
        fname, fext = os.path.splitext(self.internal_filename)
        if fext =='.gdx':
            cmdline = r'gdxdump.exe {0} symb={1} Format=csv NoHeader'.format(self.internal_filename, name)
            # cmdline = r'gdxdump.exe {0} symb={1} NoHeader Format=csv'.format(self.internal_filename, name)
            p = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
            # p = subprocess.Popen(cmdline +' | tr "[:upper:]" "[:lower:]"', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            strdata = p.communicate()[0] #.replace("'.","',").replace(', \r\n','\r\n').replace(" ",",")
            sepchar = ','
        elif fext in ['.txt','.dmp']:
            f = open (self.internal_filename, "r")
            strdata = f.read().replace("'.'",".").replace("'.",".").replace(", ","").replace(" ","' ")
            sepchar = ' '
            f.close()
        else:
            raise Exception('File extension "%s" not supported yet' % fext)
        strdata = strdata.replace('eps','1e-16')
        csvfile = StringIO.StringIO(strdata)
        # prevpos = csvfile.pos
        # datafound = False
        # for l in csvfile:
        #     if l[0] == "'":
        #         csvfile.seek(prevpos)
        #         datafound = True
        #         break
        #     prevpos = csvfile.pos
        # if not datafound:
        #     raise Exception(strdata)
        #print strdata[:500]
        try:
            df = pd.read_csv(csvfile,sep=sepchar,quotechar='"',prefix='s',header=None,error_bad_lines=False).dropna()
        except:
            pass #print 'try to go on'
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

def loadsymbols(slist,glist,gdxlabels=None,reducel=True,remove_underscore=True):
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
      print 'From GDX:\n%s' % ('\n'.join(['g%d) %s' % (ig+1,g) for ig,g in enumerate(glist)]))
      for s in slist:
            print '\n'
            sdata = {}
            svar = None
            validgdxs = []
            for ig,g in enumerate(glist):
                  if gdxlabels == None:
                      gid = 'g%d' % (ig+1)
                  else:
                      gid = gdxlabels[ig]
                  #try:
                  sdata_curr = gdxsymb(s,g).get_values()
                  # except Exception as e:
                  #     print 'WARNING: Missing "%s" from "%s"' % (s,gid)
                  #     exc_type, exc_value, exc_traceback = sys.exc_info()
                  #     traceback_details = {
                  #         'filename': exc_traceback.tb_frame.f_code.co_filename,
                  #         'lineno'  : exc_traceback.tb_lineno,
                  #         'name'    : exc_traceback.tb_frame.f_code.co_name,
                  #         'type'    : exc_type.__name__,
                  #         'message' : exc_value.message, # or see traceback._some_str()
                  #     }
                  #     print traceback_template % traceback_details
                      #exc_type, exc_obj, exc_tb = sys.exc_info()
                      #fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                      #print(exc_type, fname, exc_tb.tb_lineno)
                      #top = traceback.extract_stack()[-1]
                      #print ', '.join([type(e).__name__, os.path.basename(top[0]), str(top[1])])
                  #    continue
                  validgdxs.append(gid)
                  nax = len(sdata_curr.axes)
                  if reducel and ('l' in sdata_curr.axes[-1]):
                        nax -= 1 
                        sdata_curr = eval("sdata_curr.ix[%s,'l']" % ','.join([':' for x in range(nax)]))
                  sdata[gid] = sdata_curr
            nvg = len(validgdxs)
            if nvg == 1:
                  svar = sdata[validgdxs[0]]
            elif nvg > 1:
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
            sys.modules['__builtin__'].__dict__[s] = svar
            print s
            if isinstance(svar,pd.DataFrame):
                print svar.describe()
            else:
                print svar
            time.sleep(0.01)


