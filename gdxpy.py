import subprocess
import re
import StringIO
import pandas as pd
import sys
import time

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
        self.gdxsource = gdxsource
        self.dim = dim
        self.stype = stype
        self.desc = sdesc

    def __repr__(self):
        return '({0}) {1}'.format(self.stype,self.desc)

    def populate(self,reshape=True):
        cmdline = r'gdxdump.exe {0} symb={1} NoHeader'.format(self.gdxsource.internal_filename, self.name)
        p = subprocess.Popen(cmdline+' | tr "[:upper:]" "[:lower:]"', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        csvfile = StringIO.StringIO(p.communicate()[0].replace('eps','1e-16').replace(', \r\n','\r\n').replace("'.","',").replace(" ",","))
        prevpos = csvfile.pos
        for l in csvfile:
            if l[0] == "'":
                csvfile.seek(prevpos)
                break
            prevpos = csvfile.pos
        df = pd.read_csv(csvfile,sep=",",quotechar="'",prefix='s',header=None)
        ncols = len(df.columns)
        if ncols>1:
            df = df.pivot_table(df.columns.values[-1],rows=list(df.columns.values[:-2]),cols=df.columns.values[-2])
            if reshape and (ncols>3):
                df = convert_pivottable_to_panel(df)
        else:
            df = df.s0
        self.data = df
        return df

    def __call__(self):
        return self.populate()


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

def loadsymbols(slist,glist,reducel=True,remove_underscore=True):
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
      ng = len(glist)
      nax = 0
      print 'From GDX:\n%s' % ('\n'.join(['g%d) %s' % (ig+1,g) for ig,g in enumerate(glist)]))
      for s in slist:
            print '\n'
            sdata = {}
            svar = None
            for ig,g in enumerate(glist):
                  gid = 'g%d' % (ig+1)
                  try:
                        sdata_curr = gdxsymb(s,gdxfile(g)).populate()
                  except:
                        print 'WARNING: Missing "%s" from "%s"' % (s,gid)
                  nax = len(sdata_curr.axes)
                  if reducel and ('l' in sdata_curr.axes[-1]):
                        nax -= 1 
                        sdata_curr = eval("sdata_curr.ix[%s,'l']" % ','.join([':' for x in range(nax)]))
                  sdata[gid] = sdata_curr
            if ng == 1:
                  svar = sdata['g1']
            else:
                  if nax == 1:
                        svar = pd.DataFrame(sdata)
                  elif nax == 2:
                        svar = pd.Panel(sdata)
                  elif nax == 3:
                        svar = pd.Panel4D(sdata)
                  elif nax == 4:
                        Panel5D = pd.core.panelnd.create_nd_panel_factory(
                              klass_name   = 'Panel5D',
                              orders  = [ 'cool', 'labels','items','major_axis','minor_axis'],
                              slices  = { 'labels' : 'labels', 'items' : 'items',
                                          'major_axis' : 'major_axis', 'minor_axis' : 'minor_axis' },
                              slicer  = Panel4D,
                              aliases = { 'major' : 'major_axis', 'minor' : 'minor_axis' },
                              stat_axis    = 2)
                        svar = Panel5D(sdata)
                  else:
                        raise Exception('Dimension not supported')
            if remove_underscore:
                  s = s.replace('_','')
            sys.modules['__builtin__'].__dict__[s] = svar
            print s, svar
            time.sleep(0.01)


