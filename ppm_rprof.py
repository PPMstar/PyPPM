#
# ppm.py - Tools for accessing and visualising PPMstar data.
#          Depends on the nugridpy package developed by the
#          NuGrid collaboration
# (c) 2010 - 2013 Daniel Alexander Bertolino Conti
# (c) 2011 - 2013 Falk Herwig
# (c) 2014 - 2015 Sam Jones, Falk Herwig, Robert Andrassy
#

""" 
ppm.py

PPM is a Python module for reading Yprofile-01-xxxx.bobaaa files.
Simple session for working with ppm.py, here I assume user's working
directory contains some YProfile files.

If the user find any bugs or errors, please email us.

Yprofile files Assumptions
==========================

- labeled as YProfile-01-xxxx.bobaaa and the xxxx is the NDump that is
  located within each file.

- There can be a maximum of 9999 files in each directory.  The first 10
  lines of each file are the only place where header data is uniquely
  located.

- Header data is separated by five spaces and line breaks.

- An attribute cant be in the form of a number, ie if the user is able
  to 'float(attribute)' (in python) without an error attribute will not
  be returned.

- A row of column Names or Cycle Names is preceded and followed by a
  blank line.

- A block of data values are preceded and followed by a blank line
  except the last data block in the file, where it is not followed by a
  blank line.

- In the YProfile file, if a line of attributes contains the attribute
  Ndump, then that line of attributes and any other following lines of
  attributes in the file are cycle attributes.

Header Attribute Assumptions
============================

- Header attributes are separated by lines and instances of four spaces
  (    )

- Header attribute come in one of below here are things that do stuff
  with the data 6 types.

- The first type is the first attribute in the file. It is formatted in
  such a way that the name of the attribute is separated from its
  associated data with an equals sign ex.
  Stellar Conv. Luminosity =  1.61400E-02 x 10^43 ergs,

- The second type is when an attribute contains the sub string 'grid;'.
  It is formatted in such a way such that there are 3 numerical values
  separated by 'x' which are then followed by the string ' grid;' ex.
  384 x 384 x 384 grid;

- The third type is were name of the attribute is separated from its
  associated data with an equals sign.  Also that data may be followed
  by a unit of measurement. ex.
  Thickness (Mm) of heating shell =  1.00000E+00

- The fourth type is when an attribute contains a colon.  The String
  before the colon is the title of the attribute.  After the colon there
  is a list of n sub attributes, each with a sub title, and separated by
  its value with an equals sign.  Aslo each sub attribute is separated
  by a comma ex.
  At base of the convection zone:  R =  9.50000E+00,  g =  4.95450E-01,
  rho =  1.17400E+01,  p =  1.69600E+01

- The fifth is when an attribute starts with 'and'.  after the and, the
  next word after has to be the same as one word contained in the
  previous attribute ex.
  and of transition from convection to stability =  5.00000E-01  at
  R =  3.00000E+01 Mm.

- The sixth is when there is a string or attribute title followed by two
  spaces followed by one value followed by two spaces followed by an
  'and' which is then followed by a second Value ex.
  Gravity turns on between radii   6.00000E+00  and   7.00000E+00  Mm.

Examples
========
Here is an example runthrough.

>>> from ppm import *
>>> p=y_profile()
>>> head= p.hattri
>>> cols= p.dcols
>>> cyc=p.cattri
>>> print head
[['Stellar Conv. Luminosity', '1.61400E-02 x 10^43 ergs,'], ['384 x 384 x 384 grid;'], ... ]
>>> print cols
['j', 'Y', 'FVconv', 'UYconv', ... , 'Ek H+He']
>>> print cyc
['Ndump', 't', 'trescaled', 'Conv Ht', 'H+He Ht', ..., 'EkXZHHeMax']
>>> j2=p.getColData('j','Yprofile-01-0002',numType='file',resolution='a')
>>> print j2
[[384.0, 383.0, 382.0,..., 1.0],[192.0, 191.0, 190.0,...,1.0]]
>>> j2=p.get('j')
>>> print j2
[[384.0, 383.0, 382.0,..., 1.0],[192.0, 191.0, 190.0,...,1.0]]
>>> j55=p.getColData('j',55,numType='t',resolution='l')
The closest time is at Ndump = 2
>>> print j55
>>> y=p.getColData('Rho1 H+He',2, resolution='L')
>>> print y
[2.0420099999999999e-07, 5.4816300000000004e-07, ... , 0]

and

>>> p.plot('j','FVconv')

plots the data.

"""

# -*- coding: utf-8 -*-

from numpy import *
import numpy as np
from math import *
import nugridpy.data_plot as data_plot
from nugridpy import utils
#import utils
import matplotlib.pylab as pyl
import matplotlib.pyplot as pl
from matplotlib import rcParams
from matplotlib import gridspec
import os
import re
import nugridpy.astronomy as ast
#import astronomy as ast
import scipy.interpolate as interpolate
from scipy import optimize
import copy
import sys
#sys.path.insert(0, '/data/ppm_rpod2/lib/lcse')
#import rprofile as rprof
import struct
import logging


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('rp_info')
log.setLevel(logging.DEBUG)

center_types = [(k, 'f8') for k in ('phi', 'theta', 'x', 'y', 'z')]
normal_types = [(k, 'f8') for k in ('x', 'y', 'z')]


def prep_Yprofile_data(user="Paul", run="BW-Sakurai-1536-N13"):
    '''
    for given user and run create YProfile dir in run dir and link all 
    YProfile data from dump directories

    this is for use on BlueWaters
    '''
    import subprocess
    login_names = {'Paul': 'pwoodwar',\
                   'Stou': 'sandalsk',\
                   'Falk': 'fherwig'  }
    run_dir = '/scratch/sciteam/'+login_names[user]+'/'+run
    data_dir = run_dir+'/YProfiles'
    mkdir_command = 'mkdir '+data_dir
    subprocess.call([mkdir_command],shell=True)
    remove_broken_links = 'find -L '+data_dir+' -type l -delete'
    subprocess.call([remove_broken_links],shell=True)
    link_command = 'ln -fs '+run_dir+'/????/YProfile-01/* '+data_dir
    subprocess.call([link_command],shell=True)
    return data_dir

def set_nice_params():
    fsize=14
    
    params = {'axes.labelsize':  fsize,
    #    'font.family':       'serif',
    'font.family':        'Times New Roman',
    'figure.facecolor':  'white',
    'font.size':     fsize,
    'legend.fontsize':   fsize*0.75,
    'xtick.labelsize':   fsize*0.8,
    'ytick.labelsize':   fsize*0.8,
    'ytick.minor.pad': 8,
    'ytick.major.pad': 8,
    'xtick.minor.pad': 8,
    'xtick.major.pad': 8,
    'text.usetex':       False}
    pl.rcParams.update(params)

def set_YProf_path(path,YProf_fname='YProfile-01-0000.bobaaa'):
    '''Set path to location where YProfile directories can be found.

       For example, set path to the swj/PPM/RUNS_DIR VOSpace directory
       as a global variable, so that it need only be set once during
       an interactive session; instances can then be loaded by
       refering to the directory name that contains YProfile files.

       ppm.ppm_path  contains path
       ppm.cases     contains dirs in path that contain file with name
                     YProf_fname usually used to determine dirs with 
                     YProfile files
    '''
    global ppm_path, cases
    ppm_path = path
    cases = []
    for thing in os.listdir(ppm_path):
        dir_thing = os.path.join(ppm_path,thing)
        if os.path.isdir(dir_thing) and \
           os.path.isfile(os.path.join(ppm_path,thing,YProf_fname)):
            cases.append(thing)
            
def prof_compare(cases,ndump=None,yaxis_thing='FV H+He',ifig=None,num_type='ndump',
                 labels=None,logy=True):
    """
    Compare profiles of quantities from multiple PPM Yprofile instances at a
        given time of nump number.
        
    Parameters
    ----------
        
    cases : list
        list containing the Yprofile instances that you want to compare
    ndump : string or int, optional
        The filename, Ndump or time, if None it defaults to the
        last NDump.  The default is None.
    yaxis_thing : string, optional
        What quantity to plot on the y-axis.
        The default is 'FV H+He'
    ifig : int, optional
        Figure number. If None, chose automatically.
        The default is None.
    num_type : string, optional
        Designates how this function acts and how it interprets
        fname.  If numType is 'file', this function will get the
        desired attribute from that file.  If numType is 'NDump'
        function will look at the cycle with that nDump.  If
        numType is 'T' or 'time' function will find the _cycle
        with the closest time stamp.  The default is "ndump".
    labels : list, optional
        List of labels; one for each of the cases.
        If None, labels are simply indices.
        The default is None.
    logy : boolean, optional
        Should the y-axis have a logarithmic scale?
        The default is True.
        
    Examples
    --------
        
        import ppm
        run1='/rpod3/fherwig/PPM/RUNS_DATA/VLTP_MESA_M0.542/C1'
        run2='/rpod3/fherwig/PPM/RUNS_DATA/sakurai-num-exp-robustness-onset-GOSH/A1'
        YY=ppm.yprofile(run1)
        YY2=ppm.yprofile(run2)
        ppm.prof_compare([YY,YY2],ndump=100,num_type='time',
                        labels=['VLTP_0.543','SAK'],yaxis_thing='Rho',
                        logy=False)

    """

    fsize=14
        
    params = {'axes.labelsize':  fsize,
    #    'font.family':       'serif',
    'font.family':        'Times New Roman',
    'figure.facecolor':  'white',
    'text.fontsize':     fsize,
    'legend.fontsize':   fsize,
    'xtick.labelsize':   fsize*0.8,
    'ytick.labelsize':   fsize*0.8,
    'text.usetex':       False}
    pl.rcParams.update(params)

    jline_offset=6
    if labels is None:
        labels=[]*len(cases)
    if ifig is None:
        pl.figure()
    else:
        pl.figure(ifig)
    i=0
    for Y in cases:
        j=i+jline_offset
        if labels is None:
            labels[i] = str(i)
        Y.plot('Y',yaxis_thing,fname=ndump,numtype=num_type,legend=labels[i],\
               logy=logy,shape=utils.linestyle(j)[0],markevery=utils.linestyle(j)[1])
        i += 1

def cdiff(x):
    # compute 2nd order centred differences
    dx = (np.roll(x, -1) - np.roll(x, 1))/2.
    
    # 1st order differences to correct the boundaries
    dx[0] = x[1] - x[0]
    dx[-1] = x[-1] - x[-2]
    
    return dx

class yprofile(data_plot.DataPlot):
    """ 
    Data structure for holding data in the  YProfile.bobaaa files.
    
    Parameters
    ----------
    sldir : string
        which directory we are working in.  The default is '.'.
        
    """

    def __init__(self, sldir='.', filename_offset=0):
        """ 
        init method

        Parameters
        ----------
        sldir : string
            which directory we are working in.  The default is '.'.
        
        """

        self.files = []  # List of files in this directory
        self.cycles= []  # list of cycles in this directory
        self.hattrs = [] # header attributes
        self.dcols = []  # list of the column attributes
        self.cattrs= []  # List of the attributes of the y profiles
        self._cycle=[]    # private var
        self._top=[]      # privite var
        self.sldir = sldir #Standard Directory
        
        if not os.path.isdir(sldir): # then try if ppm_path has been set
            try:
                sldir = ppm_path+'/'+sldir
            except:
                print 'ppm_path not correctly set: '+sldir+' is not directory.'
        self.sldir = sldir
        if not os.path.isdir(sldir):  # If the path still does not exist
            print 'error: Directory, '+sldir+ ' not found'
            print 'Now returning None'
            return None
        else:
            f=os.listdir(sldir) # reads the directory
            for i in range(len(f)):  # Removes any files that are not YProfile files
                if re.search(r"^YProfile-01-[0-9]{4,4}.bobaaa$",f[i]):
                    self.files.append(f[i])
            self.files.sort()
            if len(self.files)==0: # If there are no YProfile files in the directory
                print 'Error: no YProfile named files exist in Directory'
                print 'Now returning None'
                return None
            slname=self.files[len(self.files)-1] #
            self.slname = slname
            print "Reading attributes from file ",slname
            self.hattrs,self.dcols, self._cycle=self._readFile()
            # split the header into header attributes and top attributes
            self._splitHeader()
            # return the header attributes as a dictionary
            self.hattrs=self._formatHeader()
            # returns the concatination of cycle and top attributes
            self.cattrs=self.getCattrs() 

            self.ndumpDict=self.ndumpDict(self.files, filename_offset=filename_offset)
            self.radbase = float(self.hattrs['At base of the convection zone R'])
            self.radtop  = float(self.hattrs['Thickness (Mm) of transition from convection to stability '].split()[4])
            print 'There are '+str(len(self.files))+' YProfile files in the ' +self.sldir+' directory.'
            print 'Ndump values range from '+str(min(self.ndumpDict.keys()))+' to '+str(max(self.ndumpDict.keys()))
            t=self.get('t',max(self.ndumpDict.keys()))
            t1=self.get('t',min(self.ndumpDict.keys()))
            print 'Time values range from '+ str(t1)+' to '+str(t)
            self.cycles=self.ndumpDict.keys()
        return None

    def ndumpDict(self, fileList, filename_offset=0):
        """ 
        Method that creates a dictionary of Filenames with the
        associated key of the filename's Ndump.
        
        Parameters
        ----------
        fileList : list
            A list of yprofile filenames.
            
        Returns
        -------
        dictionary
            the filenamem, ndump dictionary
            
        """
        ndumpDict={}
        for i in xrange(len(fileList)):
            ndump=fileList[i].split("-")[-1]
            ndump=int(ndump.split(".")[0])
            ndump-=filename_offset
            ndumpDict[ndump]=fileList[i]

        return ndumpDict

    def getHattrs(self):
        """ returns the list of header attributes"""
        h=self.hattrs.sorted()
        return h.sort()

    def getDCols(self):
        """ returns the list of column attributes"""
        return self.dcols

    def getCattrs(self):
        """ returns the list of cycle attributes"""
        dupe=False
        data=[]
        for i in range(len(self._cycle)):
            data.append(self._cycle[i])

        for i in range(len(self._top)): # checking for dublicate attributes
                                        # ie if an Atribute is both a top attri and a cycle attri
            for k in range(len(self._cycle)):
                if self._top[i]==self._cycle[k]:
                    dupe=True
            if not dupe:
                data.append(self._top[i])
            dupe=False

        return data

    def get(self, attri, fname=None, numtype='ndump', resolution='H', \
            silent=False, metric = None, **kwargs):
        """ 
        Method that dynamically determines the type of attribute that is
        passed into this method.  Also it then returns that attribute's
        associated data.

        Parameters
        ----------
        attri : string
            The attribute we are looking for.
        fname : string, optional
            The filename, Ndump or time, if None it defaults to the
            last NDump.  The default is None.
        numtype : string, optional
            Designates how this function acts and how it interprets
            fname.  If numType is 'file', this function will get the
            desired attribute from that file.  If numType is 'NDump'
            function will look at the cycle with that nDump.  If
            numType is 'T' or 'time' function will find the _cycle
            with the closest time stamp.  The default is "ndump".
        Resolution : string, optional
            Data you want from a file, for example if the file contains
            two different sized columns of data for one attribute, the
            argument 'a' will return them all, 'h' will return the
            largest, 'l' will return the lowest. The default is 'H'.

        """
        isCyc=False #If Attri is in the Cycle Atribute section
        isCol=False #If Attri is in the Column Atribute section
        isHead=False #If Attri is in the Header Atribute section

        if fname==None:
            fname=max(self.ndumpDict.keys())
            if not silent:
                print "Warning at yprofile.get(): fname is None, "\
                      "the last dump (%d) will be used." \
                      % max(self.ndumpDict.keys())

        if attri in self.cattrs: # if it is a cycle attribute
            isCyc = True
        elif attri in self.dcols:#  if it is a column attribute
            isCol = True
        elif attri in self.hattrs:# if it is a header attribute
            isHead = True

        # directing to proper get method
        if isCyc:                               # edit: included single = true
            return self.getCycleData(attri,fname, numtype, resolution=resolution, Single = True, \
                                     silent=silent)
        if isCol:
            return self.getColData(attri,fname, numtype, resolution=resolution, \
                                   silent=silent)
        if isHead:
            return self.getHeaderData(attri, silent=silent)
        else:
            res = computeData(attri, fname, numtype, silent=silent, **kwargs)
            if res is None:             
                if not silent:
                    print 'That Data name does not appear in this YProfile Directory'
                    print 'Returning none'
            return res 

    def getHeaderData(self, attri, silent=False):
        """ 
        Parameters
        ----------        
        attri : string
            The name of the attribute.
        
        Returns
        -------
        string or integer
            Header data that is associated with the attri.

        Notes
        -----
        To see all possable options in this instance type
        instance.getHattrs().
        
        """
        isHead = False
        if attri in self.hattrs:
            isHead = True
        if not isHead:# Error checking
            if not silent:
                print 'The attribute '+attri+' does not appear in these YProfiles'
                print 'Returning None'
            return None
        data=self.hattrs[attri] #Simple dictionary access
        return data

    def getCycleData(self, attri, FName=None, numType='ndump',
                     Single=False, resolution='H', silent=False):
        """ 
        Parameters
        ----------
        attri : string
            What we are looking for.
        FName : string, optional
            The filename, Ndump or time, if None it defaults to the last
            NDump.  The default is None.
        numType : string, optional
            Designates how this function acts and how it interprets
            FName.  If numType is 'file', this function will get the
            desired attribute from that file.  If numType is 'NDump'
            function will look at the cycle with that nDump.  If
            numType is 'T' or 'time' function will find the _cycle
            with the closest time stamp.  The default is "ndump".
        Single : boolean, optional
            Determining whether the user wants just the Attri contained
            in the specified ndump or all the dumps below that ndump.
            The default is False.
        Resolution : string, optional
            Data you want from a file, for example if the file contains
            two different sized columns of data for one attribute, the
            argument 'a' will return them all, 'h' will return the
            largest, 'l' will return the lowest.  The defalut is 'H'.

        Returns
        -------
        list
            A Datalist of values for the given attribute or a
            single attribute in the file FName.

        """

        if FName==None: #By default choose the last YProfile
            FName=max(self.ndumpDict.keys())
            if not silent:
                print "Warning at yprofile.getCycleData(): FName is None, "\
                      "the last dump (%d) will be used." % \
                      max(self.ndumpDict.keys())

        isCyc= False #If Attri is in the Cycle Atribute section
        boo=True
        filename=self.findFile(FName, numType, silent=silent)
        data=0

        if attri in self._cycle: #if attri is a cycle attribute rather than a top attribute
            isCyc = True

        if attri not in self._cycle and isCyc:# Error checking
            if not silent:
                print 'Sorry that Attribute does not appear in the fille'
                print 'Returning None'
            return None

        if not Single and isCyc:

            data= self.getColData( attri,filename,'file',resolution, True)

            return data
        #if Single and isCyc:
        #   data = self.readTop(attri,self.files[FName],self.sldir)
        if Single and isCyc:

            data= self.getColData( attri,filename,'file',resolution, True)
            #if data==None:
            #    return None
            index=len(data)-1
            return data[index]
        if not Single and not isCyc:

            data=[]
            i=0
            while boo: #Here we basically open up each YProfile File and append the required top attribute
                       # to our data variable
                data.append(self.readTop(attri,self.files[i],self.sldir))
                if self.files[i]==filename: #if we have reached the final file that the user wants.
                    boo=False
                i+=1
            for j in range(len(data)): #One top attribute hase a value of '*****' we cant float that, so we ignore it here
                if '*' not in data[j]:
                    data[j]=float(data[j])

            data=array(data)
        if Single and not isCyc:

            data=self.readTop(attri,filename,self.sldir)
            data=float(data)

        return data

    def getColData(self, attri, FName, numType='ndump', resolution='H',
                   cycle=False, silent=False):
        """ 
        Parameters
        ----------
        attri : string
            Attri is the attribute we are loking for.
        FName : string
            The name of the file, Ndump or time we are looking for.
        numType : string, optional
            Designates how this function acts and how it interprets
            FName.  If numType is 'file', this function will get the
            desired attribute from that file.  If numType is 'NDump'
            function will look at the cycle with that nDump.  If
            numType is 'T' or 'time' function will find the _cycle
            with the closest time stamp.  The default is "ndump".
        Resolution : string, optional
            Data you want from a file, for example if the file contains
            two different sized columns of data for one attribute, the
            argument 'a' will return them all, 'H' will return the
            largest, 'l' will return the lowest.  The defalut is 'H'.
        cycle : boolean, optional
            Are we looking for a cycle or column attribute.
            The default is False.

        Returns
        -------
        list
            A Datalist of values for the given attribute.

        Notes
        -----
        To ADD: options for a middle length of data.

        """
        def reduce_h(r):
            '''
            Function for reducing hi-res arrays to low-res arrays.
            To be called for FV profiles at present because of
            averaging issues.
                
            Parameters
            ----------
            r : array
                array to be reduced
                
            Output
            ------
            r : array
                reduced array
            '''
            return (r[0::2]+r[0::2])/2.
        
        num=[]       #temp list that holds the line numbers where the
                     # attribute is found in
        dataList=[]  # holds final data
        attriLine='' # hold a line that the attribute is found in
        attriList=[] # holds a list of attributes that the attribute is
                     # found in
        numList=[]   # holds a single column of data
        boo=False     #temp boolean
        tmp=''
        FName=self.findFile(FName, numType, silent=silent)

        stddir=self.sldir
        resolution=resolution.capitalize()
        if stddir.endswith('/'):  # Makeing sure that the standard dir ends with a slash
                                  # So we get something like /dir/file instead of /dirfile
            FName = str(stddir)+str(FName)
        else:
            FName = str(stddir)+'/'+str(FName)


        boo=True
        try:
            f=open(FName, 'r')
        except IOError:
            if not silent:
                print "That File, "+FName+ ", does not exist."
                print 'Returning None'
            return None
        List=f.readlines()
        f.close()
        for i in range(len(List)):#finds in what lines the attribute is
                                  #located in. This is stored in num, a list of line numbers
            tmpList=List[i].split('  ')
            for k in range(len(tmpList)):
                tmpList[k]=tmpList[k].strip()

            for k in range(len(tmpList)):
                if not cycle:
                    if attri == tmpList[k] and not('Ndump' in List[i]): #if we are looking for a column attribute
                        num.append(i)
                else:
                    if attri == tmpList[k] and ('Ndump' in List[i]): #if we are looking for a cycle attribute
                        num.append(i)


        if i==(len(List) -1) and len(num)==0: #error checking
            if not silent:
                print "Attribute DNE in file, Returning None"
            return None

        for j in range(len(num)): #for each line that attri appears in
            attriList=[]
            rowNum=num[j] #the line in the file that the attribute
                          #appears at
            attriLine=List[rowNum]

            tmpList=attriLine.split('  ') #tmplist will be a list of
                                          #attributes
            for i in range(len(tmpList)):#formating tmplist
                tmpList[i]=tmpList[i].strip()
                if tmpList[i]!='':
                    attriList.append(tmpList[i])

            for i in range(len(attriList)):
                # here we find at what element in the list attri
                # appears at, this value bacomes colNum
                if attri == attriList[i]:
                    break
            colNum=i

            rowNum+=2 #Since the line of Data is two lines after
                      #the line of attributes


            #print List, rowNum
            while rowNum<len(List) and List[rowNum]!= '\n': #and rowNum<len(List)-1:
                # while we are looking at a line with data in it
                # and not a blank line and not the last line in
                # the file
                tmpList=List[rowNum].split(None) #split the line
                                         #into a list of data
                #print tmpList, colNum
                numList.append(tmpList[colNum])
                #append it to the list of data
                rowNum+=1

            #Because an attributes column of data may appear more
            #than onece in a file, must make sure not to add it twice
            if len(dataList)==0: #No data in dataList yet, no need to check
                dataList.append(numList)

            else:
                for k in range(len(dataList)):
                    #if a list of data is allready in the
                    #dataList with the same length of numList
                    #it means the data is allready present, do not add
                    if len(numList)== len(dataList[k]):
                        boo=False

                if boo:
                    dataList.append(numList)


                boo = True

            numList=[]

        tmp=''


        tmpList=[]



        #here we format the data if the user wants higher or the lower resolution of data
        if resolution.startswith('H'):
            for i in range(len(dataList)):
                if len(dataList[i])>len(tmp):
                    tmp=dataList[i]
            dataList=array(tmp,dtype=float)
        elif resolution.startswith('L'):
            for i in range(len(dataList)):
                if len(dataList[i])<len(tmp) or len(tmp)==0:
                    tmp=dataList[i]
            dataList=array(tmp,dtype=float)
        else:
            for i in range(len(dataList)):
                for k in range(len(dataList[i])):
                    dataList[i][k]=float(dataList[i][k])
                    dataList[i][k]=float(dataList[i][k])
                dataList[i]=array(dataList[i])

            try: # If dataList is a list of lists that has one element [[1,2,3]]
            # reformat dataList as [1,2,3]
                j=dataList[1][0]
            except IndexError:
                tmp = True
            except TypeError:
                tmp = False
            if tmp:
                dataList=dataList[0]

        tmp = False

#        print resolution, len(dataList), int(self.hattrs['gridX'])
        # reduce FV arrays if resolution:
        if resolution == 'L' and len(dataList)==int(self.hattrs['gridX']):
            #print 'reducing array for low resolution request'
            dataList = reduce_h(dataList)

        return dataList
    
    def _readFile(self):
        """ 
        private routine that is not directly called by the user.
        filename is the name of the file we are reading
        stdDir is the location of filename, defaults to the
        working directory
        Returns a list of the header attributes with their values
        and a List of the column values that are located in this
        particular file and a list of directory attributes.
        
        Assumptions:
        An attribute can't be in the form of a num, if
        the user can float(attribute) without an error
        attribute will not be returned
        Lines of attributs are followd and preceded by
        *blank lines

        """
        filename = os.path.join(self.sldir,self.slname)
        f=open(filename,'r')
        line=''
        headerLines=[] # List of lines in the header section of the YProfile
        header=[]      # Single line of header data
        tmp=[]
        tmp2=[]
        headerAttri=[] # Final list of header Data to be retruned
        colAttri=[]    # Final list of column attributes to be returned
        cycAttri=[]    # Final list of cycle attributes to be returned
        for i in range(0,10): # read the first 10 lines of the YProfile
                              # Add the line to headerLines if the line is not empty
            line = f.readline()
            line=line.strip()
            if line != '':
                headerLines.append(line)

        for i in range(len(headerLines)): # For each line split on occurances of '    '
                                          # And then clean up any extra whitespace.
            header.extend(headerLines[i].split('     '))
            header[i]=header[i].strip()

        for i in range(len(header)):# for each line split up on occurances of =
            tmp=header[i].split('=')
            if len(tmp)!=2: # If there are not two parts, add the unsplit line to headerAttri
                tmp=[]
                tmp.append(header[i].strip())
                headerAttri.append(tmp)
            elif len(tmp)==2: # If there are two parts, add the list of the two parts to headerAttri
                tmp[0]=tmp[0].strip()
                tmp[1]=tmp[1].strip()
                headerAttri.append([tmp[0],tmp[1]])

        lines= f.readlines()
        boo = True
        ndump=False
        attri=[]
        for i in range(len(lines)-2): #for the length of the file
            if lines[i] =='\n'and lines[i+2]=='\n': # If there is a blank line,
                #that is followed by some line and by another blank line
                # it means the second line is a line of attributes
                line = lines[i+1] # line of attributes

                line=line.split('  ') # split it up on occurances of '  '

                for j in range(len(line)): #Clean up any excess whitespace
                    if line[j]!='':    #And add it to a list of attributes
                        attri.append(line[j].strip())

                for j in range(len(attri)):
                    """
                    if attri[j]=='Ndump':
                            i = len(lines)
                            break
                            """
                    for k in range(len(colAttri)):  #If it is not allready in the list of Attributes
                                                    # add it
                        if colAttri[k]==attri[j]:
                            boo = False
                            break
                    if boo :
                        colAttri.append(attri[j])
                    boo=True

        tmp=[]
        for i in range(len(colAttri)):#gets rid of blank lines in the list
            if colAttri[i]!='':
                tmp.append(colAttri[i])
        colAttri=tmp
        tmp=[]
        for i in range(len(colAttri)):#gets rid of numbers in the list
            try:
                float(colAttri[i])
            except ValueError:
                tmp.append(colAttri[i])
        colAttri=tmp
        tmp=[]
        # NOTE at this point in the program colAttri is a unified list of Column attributes and Cycle Attributes


        for i in range(len(colAttri)): #Here we split up our list into Column attributes and Cycle Attributes
            if colAttri[i]=='Ndump':
                # If we get to Ndump in our list of attributes, then any following attributes are cycle attributes
                ndump=True
            if not ndump:
                tmp.append(colAttri[i])
            else:
                cycAttri.append(colAttri[i])

        colAttri=tmp
        f.close()
        return headerAttri,colAttri, cycAttri

    def findFile(self, FName, numType='FILE', silent=False):
        """ 
        Function that finds the associated file for FName when Fname
        is time or NDump.
        
        Parameters
        ----------
        FName : string
            The name of the file, Ndump or time we are looking for.
        numType : string, optional
            Designates how this function acts and how it interprets
            FName.  If numType is 'file', this function will get the
            desird attribute from that file.  If numType is 'NDump'
            function will look at the cycle with that nDump.  If
            numType is 'T' or 'time' function will find the cycle with
            the closest time stamp.  The default is "FILE".
            
        """

        numType=numType.upper()
        boo=False
        indexH=0
        indexL=0
        if numType=='FILE':

            #do nothing
            return str(FName)

        elif numType=='NDUMP':
            try:    #ensuring FName can be a proper NDump, ie no letters
                FName=int(FName)
            except:
                if not silent:
                    print 'Improper value for NDump, choosing 0 instead'
                FName=0
            if FName < 0:
                if not silent:
                    print 'User Cant select a negative NDump'
                    print 'Reselecting NDump as 0'
                FName=0
            if FName not in self.ndumpDict.keys():
                if not silent:
                    print 'NDump '+str(FName)+ ' Does not exist in this directory'
                    print 'Reselecting NDump as the largest in the Directory'
                    print 'Which is '+ str(max(self.ndumpDict.keys()))
                FName=max(self.ndumpDict.keys())
            boo=True

        elif numType=='T' or numType=='TIME':
            try:    #ensuring FName can be a proper time, ie no letters
                FName=float(FName)
            except:
                if not silent:
                    print 'Improper value for time, choosing 0 instead'
                FName=0
            if FName < 0:
                if not silent:
                    print 'A negative time does not exist, choosing a time = 0 instead'
                FName=0
            keys=self.ndumpDict.keys()
            keys.sort()
            tmp=[]
            for i in xrange(len(keys)):
                timeData=self.get('t',i)
                tmp.append(timeData)

            timeData=tmp
            time= float(FName)
            for i in range(len(timeData)): #for all the values of time in the list, find the Ndump that has the closest time to FName
                if timeData[i]>time and i ==0:
                    indexH=i
                    indexL=i
                    break
                if timeData[i]>time:
                    indexH=i
                    indexL=i-1
                    break
                if i == len(timeData)-1:
                    indexH=i
                    indexL=i-1
            high=float(timeData[indexH])
            low= float(timeData[indexL])
            high=high-time
            low=time-low

            if high >=low:
                if not silent:
                    print 'The closest time is at Ndump = ' +str(keys[indexL])
                FName=keys[indexL]
            else:
                if not silent:
                   print 'The closest time is at Ndump = ' +str(keys[indexH])
                FName=keys[indexH]
            boo=True
        else:
            if not silent:
                print 'Please enter a valid numType Identifyer'
                print 'Returning None'
            return None

        if boo:#here i assume all yprofile files start like 'YProfile-01-'
            FName=self.ndumpDict[FName]
        return FName

    def _splitHeader(self):
        """ 
        Private function that splits up the data in the header section of the YProfile
        into header attributes and top attributes, where top attributes are just
        cycle attributes located in the header section of the YProfile

        """
        tmp=[]
        tmp2=[]
        slname=self.files[0]
        # Find the header section from another YProfile
        header, tm, tm1=self._readFile() 


        if len(header)!=len(self.hattrs): #error checking
            print 'Header atribute error, directory has two YProfiles that have different header sections'
            print 'Returning unchanged header'
            return None
        for i in range(len(header)):
            if header[i]==self.hattrs[i]: #if the headers are bothe the same, that means its a
                tmp.append(header[i]) #header attribute
            else:                         #Else it changes cycle to cycle and therfore
                                          #its a cycle attribute
                tmp2.append(header[i])

        for i in range(len(tmp2)):            #Formats the top attributes
            tmp2[i]=tmp2[i][0]            #Ie splits the attributes from its associated data.


        self.hattrs=tmp
        self._top=tmp2

    def _formatHeader(self):
        """ 
        Private function that takes in a set of header attributes and
        then Formats them into a dictionary.

        Input -> A List of headers in the proper format

        Assumptions
        ===========
        
        The first element in the list is Stellar Conv. Luminosity header

        The output in the dictionary looks like
        {'Stellar Conv. Luminosity':The associated data}.

        If an element contains the string 'grid;' it is the grid size
        header and the first, second and third are the x, y and z grid
        sizes respectively.

        The output in the dictionary looks like
        {'gridX':9,'gridY':9,'gridZ':9}.
        
        If an element is size two the first item is the header name and
        the second will be its associated value.

        The output in the dictionary looks like {'First Item':value}
        
        If an element contains a colon, The string preceding the colon
        is one part of the header name.  The string after the colon
        will be a list of associations in the form of the name followed
        by an equals sign followed by its value.
        
        for example a line like this would look like:

        """
        headers=self.hattrs
        dic={}
        i=0
        print "Analyzing headers ..."
        while i < len(headers):
            if i ==0: # If it is the Stellar Luminosity attribute
                tmp=headers[i][1]
                """
                if '^' in tmp[2]:


                        j=tmp[2].split('^')
                        j=float(j[0])**float(j[1])
                else:
                        j=tmp[2]
                tmp=float(tmp[0])*float(j)
                """

                dic[headers[i][0]]=tmp
                i+=1

            elif 'grid;' in headers[i][0]: # If its the grid header attribute
                tmp1=[]
                tmp= headers[i][0].split()

                for j in range(len(tmp)):
                    tmp[j]=tmp[j].strip('x')
                    tmp[j]=tmp[j].strip('')
                for j in range(len(tmp)):
                    if tmp[j]!='':
                        tmp1.append(tmp[j])
                tmp=tmp1
                dic['gridX']=tmp[0]
                dic['gridY']=tmp[1]
                dic['gridZ']=tmp[2]
                i+=1

            elif len(headers[i])==2: # If its the header attribute that is seperated by a single = sign
                tmp=headers[i][1]

                dic[headers[i][0]]=tmp
                i+=1

            elif ':' in headers[i][0]: #If its the header attribute like 'Title: a=2, b=3
                tmp=headers[i][0].split(':')
                tmp2=tmp[1].split(',')
                for j in range(len(tmp2)):
                    tmp3=tmp2[j].split('=')
                    for k in range(len(tmp3)):
                        tmp3[k]=tmp3[k].strip()
                    dic[tmp[0]+' '+tmp3[0]]=tmp3[1]
                i+=1

            elif headers[i][0].startswith('and '):

                tmp=headers[i][0].split('=',1)

                tmp2=tmp[0].lstrip('and')
                tmp2=tmp2.lstrip()
                prev=headers[i-1][0].split()
                curr=tmp2.split()
                curr=curr[0]
                tmp3=''
                for j in range(len(prev)):
                    if prev[j]== curr:
                        break;
                    tmp3+=prev[j]+' '
                tmp2=tmp3+tmp2
                dic[tmp2]=tmp[1].strip()
                i+=1

            else:
                tmp=headers[i][0].split('  ')
                for j in range(len(tmp)):
                    tmp[j]=tmp[j].strip()
                dic[tmp[0]+' Low']=tmp[1]
                dic[tmp[0]+' High']=tmp[3]
                i+=1
        return dic

# from rprofile import rprofile_reader

class rprofile(object):

  def __init__(self, source, **kwargs):
    """Load a set of RProfiles. Can pass a `path` to a set of files, or a list of `files`. Passing
    the `lazy=True` makes everything operate only from disk, without loading things into memory
    (for large datasets).

    `source` is the path to a directory containing profiles or a list of RProfile files to open

    `stride` the iteration increment. Every `stride` element is looked at during iteration. Default is 1.

    `first_dump` the first dump to iterate from or None

    `last_dump` the last dump to iterate through (inclusive)

    There is no `.get` method... you must iterate through the files like this:

    .. code-block:: python
      :linenos:

      rp_set = lcse.rprofile_set(path=targetpath, lazy=True, logging=True)
      for rp in rp_set:
        rp.get("fv_hi")
        rp.get_table("fv")
    """
    
    self.path = source if isinstance(source, str) else None
    self.files = source if isinstance(source, list) else []
    self.lazy = kwargs.get('lazy', True)
    self.stride = kwargs.get('stride', 1)
    self.first_dump = kwargs.get('first_dump')
    self.last_dump = kwargs.get('last_dump')

    self._logging = kwargs.get('logging')
    self.log = log if self._logging else None

    self.ray_profiles = {}

    self._current_ix = 0
    self._current = None

    if self.path:
      self.files = self.get_file_list_for_path(self.path) if os.path.isdir(self.path) else [self.path]

    dump_re = re.compile('(.*)-([\d]{4})\.bobaaa')

    self.dump_map = dict((int(dump_re.match(f).groups()[1]), f) for f in self.files if dump_re.match(f))
    self.file_map = dict((f, int(dump_re.match(f).groups()[1])) for f in self.files if dump_re.match(f))

    self.dumps = self.dump_map.keys()
    self.dumps.sort()

  def __iter__(self):
    self._current_ix = self.dumps.index(self.first_dump) if self.first_dump else 0
    return self

  # Python 3 bullshit
  def __next__(self):
    return self.next()

  def next(self):

    if self._current_ix < len(self.dumps):
      dump = self.dumps[self._current_ix]

      if self.last_dump and dump > self.last_dump:
        raise StopIteration()

      rp = self.ray_profiles.get(dump, rprofile_file(self.dump_map[dump], lazy=self.lazy, logging=self._logging))

      if not self.lazy and (dump not in self.ray_profiles):
          self.ray_profiles[dump] = rp

      self._current = rp
      self._current_ix += self.stride
      return rp

    else:
      raise StopIteration()

#  def reset(self):
#
#    if first_dump:
#      self.start = self.dumps.index(first_dump)
#
#    self._current_ix = self.start
#    self._current = None


  def get_file_list_for_path(self, path):
    """ Return a list of RProfiles at the given path"""

    filenames = [os.path.join(path, f) for f in os.listdir(path) if f.startswith('RProfile') and f.endswith('.bobaaa')]
    filenames.sort()

    return filenames

#  def load_files(self, filenames):
#    """ Loads `filenames` """
#
#    # This should add to the existing
#
#    self.files = filenames
#    self.files.sort()
#    self.ray_profiles = [rprofile(f, lazy=self.lazy, logging=self._logging) for f in self.files]

  def check_for_new(self, path=None):
    """Check path for new files"""

    current_files = self.get_file_list_for_path(self.path or path)

    new_files = [f for f in current_files if f not in self.files]

    self.files.extend(new_files)
    self.files.sort()

    return len(new_files) > 0
  
  #def get(self, attri, dump, globals_only = False):
    """ Get a new `rprofile` instance for `dump`. These are NOT cached internally."""
    
  def get(self, attri, fname=None, numtype='ndump', resolution='H', \
            silent=False, globals_only = True, metric = 0):
    
    dump = self.findFile(fname, numType = numtype, silent=silent)
    if self.dumps and dump is None:
      dump = self.dumps[-1]
    elif dump not in self.dump_map:
      return None
    
    rpof = self.ray_profiles.get(dump, rprofile_file(self.dump_map[dump], lazy=self.lazy, logging=self._logging)) 
    
    if attri in rpof.header_attrs:
      return rpof.header_attrs.get(attri)

    offset, dtype, count, shape = rpof._variable_map[attri]
    data = rpof.get(attri, globals_only = globals_only)
    
    if shape[0] == 4:
        data = data[metric,:]
    data = np.flip(data,0)# yprofile goes t0 -> tf rprof: tf -> t0
    return data

  def findFile(self, FName, numType='FILE', silent=False):
        """ 
        Function that finds the associated file for FName when Fname
        is time or NDump.

        Parameters
        ----------
        FName : string
            The name of the file, Ndump or time we are looking for.
        numType : string, optional
            Designates how this function acts and how it interprets
            FName.  If numType is 'file', this function will get the
            desird attribute from that file.  If numType is 'NDump'
            function will look at the cycle with that nDump.  If
            numType is 'T' or 'time' function will find the cycle with
            the closest time stamp.  The default is "FILE".

        """

        numType=numType.upper()
        boo=False
        indexH=0
        indexL=0
        if numType=='FILE':

            #do nothing
            return str(FName)

        elif numType=='NDUMP':
            try:    #ensuring FName can be a proper NDump, ie no letters
                FName=int(FName)
            except:
                if not silent:
                    print 'Improper value for NDump, choosing 0 instead'
                FName=0
            if FName < 0:
                if not silent:
                    print 'User Cant select a negative NDump'
                    print 'Reselecting NDump as 0'
                FName=0
            if FName not in self.file_map.values():
                if not silent:
                    print 'NDump '+str(FName)+ ' Does not exist in this directory'
                    print 'Reselecting NDump as the largest in the Directory'
                    print 'Which is '+ str(max(self.file_map.values()))
                FName=max(self.file_map.values())
            boo=True

        elif numType=='T' or numType=='TIME':
            try:    #ensuring FName can be a proper time, ie no letters
                FName=float(FName)
            except:
                if not silent:
                    print 'Improper value for time, choosing 0 instead'
                FName=0
            if FName < 0:
                if not silent:
                    print 'A negative time does not exist, choosing a time = 0 instead'
                FName=0
            dump = self.file_map[max(self.file_map.keys())]
            #timeData=self.get('time',self.file_map[max(self.file_map.keys())],numtype='file')
            keys=self.file_map.values()
            keys.sort()
            tmp=[]
            for i in xrange(len(keys)):
                rpof = self.ray_profiles.get(i,\
                   rprofile_file(self.dump_map[i], lazy=self.lazy, logging=self._logging)) 
                timeData = rpof.get('t', True)
                tmp.append(timeData)
            timeData=tmp
            time= float(FName)
            for i in range(len(timeData)): #for all the values of time in the list, find the Ndump that has the closest time to FName
                if timeData[i]>time and i ==0:
                    indexH=i
                    indexL=i
                    break
                if timeData[i]>time:
                    indexH=i
                    indexL=i-1
                    break
                if i == len(timeData)-1:
                    indexH=i
                    indexL=i-1
            high=float(timeData[indexH])
            low= float(timeData[indexL])
            high=high-time
            low=time-low

            if high >=low:
                if not silent:
                    print 'The closest time is at Ndump = ' +str(keys[indexL])
                FName=keys[indexL]
            else:
                if not silent:
                    print 'The closest time is at Ndump = ' +str(keys[indexH])
                FName=keys[indexH]
            boo=True
        else:
            if not silent:
                print 'Please enter a valid numType Identifyer'
                print 'Returning None'
            return None

        #if boo:#here i assume all yprofile files start like 'YProfile-01-'
        #    FName=self.file_map[FName]
        return FName

class rprofile_file(object):
  """
   `rprofile.header_attrs` is a dictionary of header attributes

  """


  header_var_list = [
      dict(name='version', pos=0, type='i'),
      dict(name='cell_ct_low', pos=1, type='i'),
      dict(name='nbuckets', pos=2, type='i'),
      dict(name='dump', pos=3, type='i'),
      dict(name='sizeof_float', pos=4, type='i'),
      dict(name='has_centers', pos=5, type='i'),
      dict(name='has_corners', pos=6, type='i'),
      dict(name='has_normals', pos=7, type='i'),
      dict(name='isrestart', pos=8, type='i', min_ver=12),
      dict(name='var_ct_low', pos=9, type='i'),
      dict(name='var_ct_high', pos=10, type='i'),
      dict(name='cell_ct_high', pos=11, type='i'),
      dict(name='ncpucores', pos=12, type='i'),
      dict(name='ntxbricks', pos=13, type='i'),
      dict(name='ntybricks', pos=14, type='i'),
      dict(name='ntzbricks', pos=15, type='i'),
      dict(name='nxteams', pos=16, type='i'),
      dict(name='nyteams', pos=17, type='i'),
      dict(name='nzteams', pos=18, type='i'),
      dict(name='nx', pos=19, type='i'),
      dict(name='ny', pos=20, type='i'),
      dict(name='nz', pos=21, type='i'),
      dict(name='nsugar', pos=22, type='i'),
      dict(name='nbdy', pos=23, type='i'),
      dict(name='nfluids', pos=24, type='i'),
      dict(name='nvars', pos=25, type='i'),
      dict(name='nhalfwaves', pos=26, type='i'),
      dict(name='maxrad', pos=27, type='i'),
      dict(name='nteamsinbunch', pos=28, type='i'),
      dict(name='ndumps', pos=29, type='i'),
      dict(name='ndumpstodo', pos=30, type='i'),
      dict(name='nrminrad', pos=31, type='i'),
      dict(name='nrmaxrad', pos=32, type='i'),
      dict(name='iburn', pos=33, type='i', min_ver=12),
      dict(name='imuffledbdry', pos=34, type='i', min_ver=12),
      dict(name='ireflectbdry', pos=35, type='i', min_ver=12),

      # fheader (the offsets are in the fheader)
      dict(name='radin0', pos=0, type='f', help='Gravity completely off inside this radius'),
      dict(name='radinner', pos=1, type='f', help='Gravity starts turning off inside this radius'),
      dict(name='radbase', pos=2, type='f', help='Bot convect zone'),
      dict(name='radtop', pos=3, type='f', help='Top convect zone'),
      dict(name='radouter', pos=4, type='f', help='Grav starts turning off outside this radius'),
      dict(name='radout0', pos=5, type='f', help='Gravity completely off outside this radius'),
      dict(name='radmax', pos=6, type='f', help='distance from center of grid to nearest edge'),
      dict(name='dlayerbot', pos=7, type='f', help='thickness of flame zone'),
      dict(name='dlayertop', pos=8, type='f', help='thickness of transition @ top of convect zone'),
      dict(name='totallum', pos=9, type='f'),
      dict(name='grav00base', pos=10, type='f'),
      dict(name='rho00base', pos=11, type='f'),
      dict(name='prs00base', pos=12, type='f'),
      dict(name='gammaconv', pos=13, type='f'),
      dict(name='gammabelow', pos=14, type='f'),
      dict(name='gammaabove', pos=15, type='f'),
      dict(name='gravconst', pos=16, type='f'),
      dict(name='rhoconv', pos=17, type='f'),
      dict(name='rhoabove', pos=18, type='f'),
      dict(name='airmu', pos=19, type='f', min_ver=13),
      dict(name='cldmu', pos=20, type='f', min_ver=13),
      dict(name='fkair', pos=21, type='f', min_ver=13),
      dict(name='fkcld', pos=22, type='f', min_ver=13),
      dict(name='atomicnoair', pos=23, type='f', min_ver=13),
      dict(name='atomicnocld', pos=24, type='f', min_ver=13),

      # Global T-history
      dict(name='t', pos=31+0, type='f'),
      dict(name='timerescaled', pos=31+1, type='f'),
      dict(name='bubbleheight', pos=31+2, type='f'),
      dict(name='spikeheight', pos=31+3, type='f'),
      dict(name='cycl', pos=31+4, type='f', min_ver=12),
      dict(name='dt', pos=31+5, type='f'),
      dict(name='courmx', pos=31+6, type='f'),
      dict(name='urbubmx', pos=31+7, type='f'),
      dict(name='urspkmn', pos=31+8, type='f'),
      dict(name='ekmx', pos=31+9, type='f'),
      dict(name='ekrmx', pos=31+10, type='f'),
      dict(name='ektmx', pos=31+11, type='f'),
      dict(name='ekurmn', pos=31+12, type='f'),
      dict(name='ekurmx', pos=31+13, type='f'),
      dict(name='eiurmn', pos=31+14, type='f'),
      dict(name='eiurmx', pos=31+15, type='f'),
      dict(name='Hurmn', pos=31+16, type='f'),
      dict(name='Hurmx', pos=31+17, type='f'),
      dict(name='ekurspkmn', pos=31+18, type='f'),
      dict(name='ekurbubmx', pos=31+19, type='f'),
      dict(name='eiurspkmn', pos=31+20, type='f'),
      dict(name='eiurbubmx', pos=31+21, type='f'),
      dict(name='Hurspkmn', pos=31+22, type='f'),
      dict(name='Hurbubmx', pos=31+23, type='f'),
      dict(name='ekbubmx', pos=31+24, type='f'),
      dict(name='ekrbubmx', pos=31+25, type='f'),
      dict(name='ektbubmx', pos=31+26, type='f'),
      dict(name='ekspkmx', pos=31+27, type='f'),
      dict(name='ekrspkmx', pos=31+28, type='f'),
      dict(name='ektspkmx', pos=31+29, type='f'),

      # Args images
      dict(name='ai_vort', pos=64+0, type='f', len=2),
      dict(name='ai_divu', pos=64+2, type='f', len=2),
      dict(name='ai_s', pos=64+4, type='f', len=2),
      dict(name='ai_fv', pos=64+6, type='f', len=2),
      dict(name='ai_rho', pos=64+8, type='f', len=2),
      dict(name='ai_p', pos=64+10, type='f', len=2),
      dict(name='ai_ux', pos=64+12, type='f', len=2),
      dict(name='ai_uy', pos=64+14, type='f', len=2),
      dict(name='ai_uz', pos=64+16, type='f', len=2),
  ]

  def __init__(self, filename, lazy=True, **kwargs):
    """Create a ray profile reader object.

    `lazy` means only the header is loaded on open
    """

    logging = kwargs.get('logging')

    self._filename = filename
    self.lazy = lazy

    self.version = None
    self.bucket_count = 0

    self._centers = None
    self._corners = None
    self._normals = None

    self._cache = {}

    self._variable_map = {}

    self._names = []
    self._data = []

    self.header_attrs = {}

    if logging:
      import logging
      logging.basicConfig(level=logging.DEBUG)
      self.log = logging.getLogger('rp_info')
      self.log.setLevel(logging.DEBUG)
    else:
      self.log = None

    if str(filename).isdigit():
      filename = 'RProfile-01-%04i.bobaaa' % int(filename)

    if self.log: self.log.info("Opening %s" % filename)

    f = open(filename, 'rb')
    header = f.read(128)

    if header[:8] != 'LCSE:RPS':
      if self.log: self.log.warn('File %s is not a new Ray Profile, try an older rp_info.py' % filename)
      f.close()
      raise Exception('Unsupported file version')

    self.version = struct.unpack("i", header[8:12])[0]

    f.seek(0)

    if self.version < 8:
      raise Exception('Unsupported version %i' % self.version)
    elif self.version == 8:
      self._header_size = 128
      hlen = 8

      self.header_var_list = self.header_var_list[:8]

#      header = struct.unpack(hlen * "i", header[8:8+4*hlen])

#      self.header_attrs['version'] = header[0]
#      self.header_attrs['cell_ct_low'] = header[1]
#      self.header_attrs['nbuckets'] = header[2]
#      self.header_attrs['dump'] = header[3]
#      self.header_attrs['sizeof_float'] = header[4]
#      self.header_attrs['has_centers'] = header[5]
#      self.header_attrs['has_corners'] = header[6]
#      self.header_attrs['has_normals'] = header[7]

    elif self.version > 8:
      self._header_size = 1024
      hlen = 127

    header = f.read(self._header_size)
      # Bug fixes
      # Using the header info from v9
     # if self.version < 11:
     #   self._init_v9()
     #   self._init_legacy()

     #   raw_header = struct.unpack(hlen * "i", header[8:8+4*hlen])
     #   raw_fheader = struct.unpack(hlen * "f", header[8+4*hlen:8+8*hlen])
     #   self.header_attrs.update([(k, raw_header[i]) for i, k in enumerate(self._header_names)])
     #   self.header_attrs.update([(k, raw_fheader[i]) for i, k in enumerate(self._fheader_names)])
     #   self.header_attrs.update([(k, raw_fheader[32 + i]) for i, k in enumerate(self._fheader_names2)])
     #   self.header_attrs.update([(k, (raw_fheader[64 + 2*i], raw_fheader[64 + 2*i + 1] )) for i, k in enumerate(self._argsimg_names)])
     #elif self.version <= 12:

    hmap = dict(i=struct.unpack(hlen * "i", header[8 : 8 + 4 * hlen]),
                f=struct.unpack(hlen * "f", header[8 + 4 * hlen : 8 * (1 + hlen)]))

    for var in self.header_var_list:

      name = var['name']
      pos = var['pos']
      var_type = var['type']
      var_len = var.get('len', 1)
      min_ver = var.get('min_ver', 0)

      if self.version < min_ver:
        continue

      # A slight offset problem
      if self.version == 11 and var_type == 'f' and pos > 30: # and pos < 64:
        pos = pos + 1

      attr = hmap[var_type][pos] if var_len == 1 else hmap[var_type][pos : pos + var_len]
      self.header_attrs[name] = attr

    # Fix header problems
    if self.version == 8:
      self.header_attrs['cell_ct_high'] = 2 * self.header_attrs['cell_ct_low']
      self.header_attrs['var_ct_high'] = 1
      self.header_attrs['var_ct_low'] = 14

    if self.version == 9:
      self.header_attrs['cell_ct_low'] -= 2
      self.header_attrs['cell_ct_high'] -= 4

    if self.version < 12:
      self.header_attrs['isreflectbdry'] = 1
      self.header_attrs['ismuffledbdry'] = 0

    if self.version > 13:
      self.header_attrs['has_corners'] = False

    self.bucket_count = self.header_attrs['nbuckets']
    self.dump = self.header_attrs['dump']
    self.buckets = self.header_attrs['nbuckets']

    if self.version > 10:
      self._init_v11()

      if not self.lazy:
        f = open(self._filename, 'r')
        self._data = f.read()
        f.close()

    else:
      self._init_legacy()

      if self.version == 8:
        self._init_v8()
      else:
        self._init_v9()

      for k in ['has_centers', 'has_corners', 'has_normals']:
        self.header_attrs[k] = self.header_attrs.get(k, 0) == 1

      float_type = 'f8' if self.header_attrs.get('sizeof_float') == 8 else 'f4'

      self._dtypes_hi = [('j_hi', 'i4')]
      self._dtypes_hi.extend([(n, float_type) for n in self._names_hi])
      self._col_names_hi = ['j_hi'] + self._names_hi

      self._dtypes = [('j', 'i4')]
      self._dtypes.extend([(n, float_type) for n in self._names])

      self._col_names = ['j'] + self._names

      if self.lazy:
        log.warn("Lazy Loading not supported for v %i" % self.version)

      self._load(f)

    f.close()

  def _load(self, f):

    nbuckets = self.header_attrs.get('nbuckets')
    cell_ct_low = self.header_attrs.get('cell_ct_low')
    cell_ct_high = self.header_attrs.get('cell_ct_high')

    # Read the high resolution table
    self._data_hi = np.fromfile(f, dtype=self._dtypes_hi, count=cell_ct_high*(nbuckets+1))

    # Read the low resolution table
    self._data_low = np.fromfile(f, dtype=self._dtypes, count=cell_ct_low*(nbuckets+1))

    if self.header_attrs.get('has_centers'):
      vals = 3 if self.version > 12 else 5
      self._centers = np.fromfile(f, dtype=np.float64, count=5 * nbuckets).reshape((vals, -1), order='F')

    if self.header_attrs.get('has_normals'):
      self._normals = np.fromfile(f, dtype=np.float64, count=9*nbuckets).reshape((3, 3, -1), order='F')

    if self.header_attrs.get('has_corners'):
      self._corners = np.fromfile(f, dtype=np.float64, count=9*nbuckets).reshape((3, 3, -1), order='F')
 
    
  def get(self, var, globals_only = False):
    """Get the global bucket for variable `var` or  get header attribute `var`.

    Use `get_table(self, var)` to get the same variable but for all buckets.

    If the global bucket is returned an array of dimension (4, ncells) is returned.
    The first dimension contains avg, min, max, sd.
    """

    if var in self.header_attrs:
      return self.header_attrs.get(var)

    if var in self._variable_names:
      return self._get_array(var, global_only=True)
    
  def _get_array(self, var, global_only):

    if var not in self._variable_map:
      return None

    if var in self._cache:
      return self._cache[var]

    offset, dtype, count, shape = self._variable_map[var]

#    print self._variable_map[var], global_only

    if global_only and len(shape) == 3 and shape[2] == self.bucket_count + 1:
      count = shape[0] * shape[1]
      shape = shape[:2] #used to be shape[:2]

    if self.lazy:
      f = open(self._filename, 'r')
      f.seek(offset)
      data = np.fromfile(f, dtype=dtype, count=count).reshape(shape, order='F')
      f.close()
    else:
      data = np.frombuffer(self._data[offset:], dtype=dtype,
                           count=count).reshape(shape, order='F')

    if not global_only:
      self._cache[var] = data

    return data
  def get_variables(self):
    return self._variable_names

  def _init_v11(self):

    cell_ct_high = self.header_attrs.get('cell_ct_high')
    cell_ct_low = self.header_attrs.get('cell_ct_low')

    buckets_total = 1 + self.bucket_count

    sizeof_float = self.header_attrs.get('sizeof_float')
    float_type = np.float64 if sizeof_float == 8 else np.float32
    int_type = np.int32

    vals = 3 if self.version > 12 else 5

    # name, size_in_bytes, <array dimensions>
    '''
    self._variable_list = [('centers', float_type, sizeof_float, (vals, self.bucket_count)),
                           ('normals', float_type, sizeof_float, (3, 3, self.bucket_count)),
                           ('corners', float_type, sizeof_float, (3, 3, self.bucket_count)),
                           ('j_hi', int_type, 4, (cell_ct_high,)),
                           ('y_hi', float_type, sizeof_float, (cell_ct_high,)),
                           ('fv_hi', float_type, sizeof_float, (4, cell_ct_high, buckets_total)),
                           ('j', int_type, 4, (cell_ct_low,)),
                           ('Y', float_type, sizeof_float, (cell_ct_low,)),
                           ('counts', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('fv', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('Rho', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('rhobubble', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('rhospike', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('rhourbubble', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('rhourspike', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('P', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('ux', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('uy', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('uz', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('ceul', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('mach', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('enuc', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('fnuc', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('dy', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('Ekr', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('ekt', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('Ek', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('EkUr', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('EiUr', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('HUr', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                          ]
     '''
    self._variable_list = [('centers', float_type, sizeof_float, (vals, self.bucket_count)),
                           ('normals', float_type, sizeof_float, (3, 3, self.bucket_count)),
                           ('corners', float_type, sizeof_float, (3, 3, self.bucket_count)),
                           ('j_hi', int_type, 4, (cell_ct_high,)),
                           ('y_hi', float_type, sizeof_float, (cell_ct_high,)),
                           ('fv_hi', float_type, sizeof_float, (4, cell_ct_high, buckets_total)),
                           ('j', int_type, 4, (cell_ct_low,)),
                           ('Y', float_type, sizeof_float, (cell_ct_low,)),
                           ('counts', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('fv', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('Rho', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('rhobubble', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('rhospike', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('rhourbubble', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('rhourspike', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('P', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('ux', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('uy', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('uz', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('ceul', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('mach', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('enuc', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('fnuc', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('dy', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('EkY', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('ekt', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('Ek', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('EkUr', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('EiUr', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                           ('HUr', float_type, sizeof_float, (4, cell_ct_low, buckets_total)),
                          ]
#    present_vars = [v['name'] for v in self.header_var_list if self.version >= v.get('min_ver', self.version)]

    # TODO: HACK UGH
    skip_vars = []
    if self.version > 12:
      skip_vars.append('corners')

    offset = self._header_size

    for name, dtype, sizeof, shape in self._variable_list:

      if name in skip_vars:
        continue

#      print (name, offset, dtype, sizeof, shape)

      count = np.prod(shape)
      size = sizeof * count
      self._variable_map[name] = (offset, dtype, count, shape)
      offset += size
#      print (name, offset, dtype, count, shape, sizeof)

    self._variable_names = self._variable_map.keys()
    self._variable_names.sort()

def computeData( attri, fname, numtype = 'ndump', silent=False,\
                **kwargs):
    def get_missing_args(required_args, **kwargs):
        missing_args = []

        for this_arg in required_args:
            if not this_arg in kwargs:
                missing_args.append(this_arg)

        return missing_args

    # The unit of G in the code is 10^{-3} g cm^3 s^{-2}.
    G_code = ast.grav_const/1e-3

    nabla_ad = 0.4

    if attri == 'T9':
        required_args = ('airmu', 'cldmu')
        missing_args = get_missing_args(required_args, **kwargs)
        if len(missing_args) > 0:
            if not silent:
                print 'The following arguments are missing: ', \
                      missing_args
            return None

        airmu = kwargs['airmu']
        cldmu = kwargs['cldmu']

        if 'T9_corr_params' in kwargs:
            T9_corr_params = kwargs['T9_corr_params']
        else:
            T9_corr_params = None

        rho = self.get('Rho', fname, numtype = numtype, resolution = 'l', \
                       silent = silent)
        p = self.get('P', fname, numtype = numtype, resolution = 'l', \
                     silent = silent)
        fv = self.get('FV H+He', fname, numtype = numtype, resolution = 'l', \
                      silent = silent)

        mu = fv*cldmu + (1. - fv)*airmu

        # gas constant in code units
        RR = 8.3144598

        T9 = mu*p/(RR*rho)

        if T9_corr_params is not None:
            T9 = T9_corr_params[0]*T9**T9_corr_params[1]

        return T9
    elif attri == 'Hp':
        r = self.get('Y', fname, numtype, resolution = 'l', silent = silent)
        p = self.get('P', fname, numtype, resolution = 'l', silent = silent)

        logp = np.log(p)

        dr = cdiff(r)
        dlogp = cdiff(logp)

        Hp = -dr/dlogp

        return Hp
    elif attri == 'g':
        r = self.get('Y', 0., 'time', resolution = 'l')
        r_bot = float(self.get('At base of the convection zone R', 0))
        rho = self.get('Rho', 0., 'time', resolution = 'l')
        g_bot = float(self.get('At base of the convection zone g', 0))

        # Centre r_bot on the nearest cell.
        idx_bot = np.argmin(np.abs(r - r_bot))

        # Mass below r_bot (using r[idx_bot] instead of r_bot makes 
        # g[idx_bot] == g_bot).
        m_bot = g_bot*(r[idx_bot]**2)/G_code

        dm = -4.*np.pi*(r**2)*cdiff(r)*rho
        m = -np.cumsum(dm) # Get the mass profile by integration.

        # Shift the mass profile to make sure that m[idx_bot] == m_bot.
        # The mass profile at small radii won't make sense, because
        # the core is artificial with no gravity.
        m += m_bot - m[idx_bot]

        g = G_code*m/(r**2) # Gravity profile (see the note above).

        return g
    elif attri == 'nabla':
        T9 = self.get('T9', fname, numtype, resolution = 'l', silent = silent, \
                      **kwargs)
        p = self.get('P', fname, numtype, resolution = 'l', silent = silent)

        logT9 = np.log(T9)
        logp = np.log(p)

        dlogT9 = cdiff(logT9)
        dlogp = cdiff(logp)

        nabla = dlogT9/dlogp

        return nabla
    elif attri == 'nabla_rho':
        rho = self.get('Rho', fname, numtype, resolution = 'l', silent = silent)
        p = self.get('P', fname, numtype, resolution = 'l', silent = silent)

        logrho = np.log(rho)
        logp = np.log(p)

        dlogrho = cdiff(logrho)
        dlogp = cdiff(logp)

        nabla_rho = dlogrho/dlogp

        return nabla_rho
    elif attri == 'N2':
        g = self.get('g', 0, resolution = 'l')
        Hp = self.get('Hp', fname, numtype, resolution = 'l', silent = silent)
        nabla_rho = self.get('nabla_rho', fname, numtype, resolution = 'l', silent = silent)

        # Ideal gas assumed.
        N2 = (g/Hp)*(nabla_ad - 1. + nabla_rho)

        return N2
    elif attri == 'enuc_C12pg':
        required_args = ('airmu', 'cldmu', 'fkair', 'fkcld', \
                         'AtomicNoair', 'AtomicNocld')
        missing_args = get_missing_args(required_args, **kwargs)
        if len(missing_args) > 0:
            if not silent:
                print 'The following arguments are missing: ', \
                missing_args
            return None

        airmu = kwargs['airmu']
        cldmu = kwargs['cldmu']
        fkair = kwargs['fkair']
        fkcld = kwargs['fkcld']
        AtomicNoair = kwargs['AtomicNoair']
        AtomicNocld = kwargs['AtomicNocld']

        kwargs2 = {}
        if 'Q' in kwargs:
            kwargs2['Q'] = kwargs['Q']

        if 'corr_fact' in kwargs:
            kwargs2['corr_fact'] = kwargs['corr_fact']

        if 'use_dt' in kwargs:
            kwargs2['use_dt'] = kwargs['use_dt']

        if 'T9_corr_params' in kwargs:
            kwargs2['T9_corr_params'] = kwargs['T9_corr_params']

        enuc_C12pg = self._get_enuc_C12pg(fname, airmu, cldmu, fkair, \
                                          fkcld, AtomicNoair, AtomicNocld, \
                                          numtype = numtype, silent = silent, \
                                          **kwargs2)

        return enuc_C12pg
    elif attri == 'L_C12pg':
        enuc_C12pg = self.get('enuc_C12pg', fname, numtype, resolution = 'l', \
                              silent = silent, **kwargs)
        r = self.get('Y', fname, numtype, resolution = 'l', silent = silent)

        dV = -4.*np.pi*r**2*cdiff(r)
        L_C12pg = np.sum(enuc_C12pg*dV)

        return L_C12pg
    elif attri == 'enuc_C12C12':
        required_args = ('airmu', 'cldmu', 'fkcld', 'AtomicNocld')
        missing_args = get_missing_args(required_args, **kwargs)
        if len(missing_args) > 0:
            if not silent:
                print 'The following arguments are missing: ', \
                missing_args
            return None

        airmu = kwargs['airmu']
        cldmu = kwargs['cldmu']
        fkcld = kwargs['fkcld']
        AtomicNocld = kwargs['AtomicNocld']

        kwargs2 = {}
        if 'Q' in kwargs:
            kwargs2['Q'] = kwargs['Q']

        if 'corr_fact' in kwargs:
            kwargs2['corr_fact'] = kwargs['corr_fact']

        if 'corr_func' in kwargs:
            kwargs2['corr_func'] = kwargs['corr_func']

        if 'T9_func' in kwargs:
            kwargs2['T9_func'] = kwargs['T9_func']

        enuc_C12C12 = self._get_enuc_C12C12(fname, airmu, cldmu, \
                      fkcld, AtomicNocld, numtype=numtype, \
                      silent=silent, **kwargs2)

        return enuc_C12C12
    elif attri == 'enuc_O16O16':
        required_args = ('airmu', 'cldmu', 'XO16conv')
        missing_args = get_missing_args(required_args, **kwargs)
        if len(missing_args) > 0:
            if not silent:
                print 'The following arguments are missing: ', \
                missing_args
            return None

        airmu = kwargs['airmu']
        cldmu = kwargs['cldmu']
        XO16conv = kwargs['XO16conv']

        kwargs2 = {}
        if 'Q' in kwargs:
            kwargs2['Q'] = kwargs['Q']

        if 'corr_fact' in kwargs:
            kwargs2['corr_fact'] = kwargs['corr_fact']

        if 'T9_func' in kwargs:
            kwargs2['T9_func'] = kwargs['T9_func']

        enuc_O16O16 = self._get_enuc_O16O16(fname, airmu, cldmu, \
                      XO16conv, numtype=numtype, silent=silent, \
                      **kwargs2)

        return enuc_O16O16
    else:
        return None 

def _get_enuc_C12pg( fname, airmu, cldmu, fkair, fkcld, \
                    AtomicNoair, AtomicNocld, numtype = 'ndump', \
                    Q = 1.944, corr_fact = 1., use_dt = False, \
                    T9_corr_params = None, silent = False):
    T9 = self.get('T9', fname = fname, numtype = numtype, \
                  resolution = 'l', airmu = airmu, cldmu = cldmu, \
                  T9_corr_params = T9_corr_params, silent = silent)
    fv = self.get('FV H+He', fname = fname, numtype = numtype, \
                  resolution = 'l', silent = silent)
    rho = self.get('Rho', fname = fname, numtype = numtype, \
                   resolution = 'l', silent = silent)
    rhocld = self.get('Rho H+He', fname = fname, numtype = numtype, \
                      resolution = 'l', silent = silent)
    rhoair = self.get('RHOconv', fname = fname, numtype = numtype, \
                      resolution = 'l', silent = silent)
    dt = float(self.get('dt', fname = fname, numtype = numtype, \
                  silent = silent))

    TP13 = T9**(1./3.)
    TP23 = TP13*TP13
    TP12 = np.sqrt(T9)
    TP14 = np.sqrt(TP12)
    TP32 = T9*TP12
    TM13 = 1./TP13
    TM23 = 1./TP23
    TM32 = 1./TP32

    T9inv = 1. / T9
    thyng = 2.173913043478260869565 * T9
    vc12pg = 20000000.*TM23 * np.exp(-13.692*TM13 - thyng*thyng)
    vc12pg = vc12pg * (1. + T9*(9.89-T9*(59.8 - 266.*T9)))
    thing2 = vc12pg + TM32*(1.0e5 * np.exp(-4.913*T9inv) + \
                            4.24e5 * np.exp(-21.62*T9inv))

    thing2[np.where(T9 < .0059)] = 0.
    thing2[np.where(T9 > 0.75)] = 200.

    vc12pg = thing2 * rho * 1000.

    v = 1./ rho
    atomicnocldinv = 1./AtomicNocld
    atomicnoairinv = 1./AtomicNoair

    Y1 =  rhocld * fv * v * atomicnocldinv
    Y2 =  rhoair * (1. - fv) * v * atomicnoairinv

    smaller = .0000001
    reallysmall = smaller * smaller    
    CN = 96.480733
    if use_dt:
        # We want the average rate during the current time step.
        # If the burning is too fast all the stuff available burns
        # in a fraction of the time step. We do not allow to burn 
        # more than what is available, so the average burn rate is
        # lower than then instantaneous one.
        thing3 = fkair * Y1 * Y2 * vc12pg * dt
        thing3[where(Y1 < reallysmall)] = 0.
        thing2 = np.min(np.array((thing3, Y1)), axis = 0)

        #for i in range(len(Y1)):
        #    print '{:d}   {:.1e}   {:.1e}   {:.1e}'.format(i, Y1[i], thing3[i], Y1[i]/thing3[i])

        DY = fkcld * thing2
        enuc = DY * rho * CN * Q / dt
    else:
        # We want the instantaneous burning rate. This does not
        # depend on how much stuff is available.
        thing3 = fkair * Y1 * Y2 * vc12pg

        DY = fkcld * thing3
        enuc = DY * rho * CN * Q

    # This factor can account for the heating bug if present.
    enuc *= corr_fact

    return enuc

def _get_enuc_C12C12( fname, airmu, cldmu, fkcld, AtomicNocld,\
                    numtype='ndump', silent=False, Q=9.35, \
                    corr_fact=1., corr_func=None, T9_func=None):
    if T9_func is None:
        print 'Corrected T9 profile not supplied, using uncorrected T9.'
        T9 = self.get('T9', fname=fname, numtype=numtype, \
                      resolution='l', airmu=airmu, cldmu=cldmu, \
                      silent=silent)
    else:
        T9 = T9_func(self, fname=fname, numtype=numtype, resolution='l')

    fv = self.get('FV H+He', fname=fname, numtype=numtype, \
                  resolution='l', silent=silent)
    rho = self.get('Rho', fname=fname, numtype=numtype, \
                   resolution='l', silent=silent)
    rhocld = self.get('Rho H+He', fname=fname, numtype=numtype, \
                      resolution='l', silent=silent)

    TP13 = T9**(1./3.)
    TP23 = TP13*TP13
    thyng = np.sqrt(T9)
    TP14 = np.sqrt(thyng)
    TP12 = TP14*TP14
    TP32 = T9*TP12
    TM13 = 1./TP13
    TM23 = 1./TP23
    TM32 = 1./TP32

    T9B = T9/(1.+0.0396*T9)
    T9B13 = T9B**(1./3.)
    T9B56 = T9B**(5./6.)
    T932 = T9**(3./2.)

    # C12(C12,a)ne20,60, VITAL
    BRCCA = 0.65*np.ones(len(T9))

    # BRCCN according to Dayras et al. 1977 NUC. PHYS. A 279
    BRCCN = np.zeros(len(T9))

    tmp = -(0.766e0/T9**3.e0)
    tmp = (1.e0 + 0.0789e0*T9 + 7.74e0*T9**3.e0)*tmp
    tmp = 0.859e0*np.exp(tmp)
    idx = where((T9 >= 0.5) & (T9 <= 1.5))
    BRCCN[idx] = tmp[idx]

    idx = where((T9 > 1.5) & (T9 <= 5.0))
    BRCCN[idx] = 0.055e0*(1.e0 - np.exp(-(0.789e0*T9[idx] - 0.976e0)))

    idx = where(T9 > 5.0)
    BRCCN[idx] = 0.02e0

    # Rate from CF88 for C12+C12 (MG24):
    C12C12 = 4.27e26*T9B56/T932*np.exp(-84.165/T9B13-2.12e-03*T9**3)

    # Channel modifcation to get (c12,a):
    vc12c12 = (C12C12 - C12C12*BRCCN)*BRCCA

    vc12c12 = vc12c12 * rho * 1000.

    v = 1./ rho
    atomicnocldinv = 1./AtomicNocld

    Y1 =  rhocld * fv * v * atomicnocldinv

    thing2 = fkcld * Y1 * Y1 * vc12c12

    DY = 0.5 * fkcld * thing2

    CN = 96.480733
    enuc = DY * rho * CN * Q

    # This factor can account for the heating bug if present.
    enuc *= corr_fact

    if corr_func is not None:
        cf = corr_func(self, fname=fname, numtype=numtype, resolution='l')
        enuc *= cf

    return enuc

def _get_enuc_O16O16( fname, airmu, cldmu, XO16conv, \
                     numtype='ndump', silent=False, corr_fact=1., \
                     T9_func=None):
    if T9_func is None:
        print 'Corrected T9 profile not supplied, using uncorrected T9.'
        T9 = self.get('T9', fname=fname, numtype=numtype, \
                      resolution='l', airmu=airmu, cldmu=cldmu, \
                      silent=silent)
    else:
        T9 = T9_func(self, fname=fname, numtype=numtype, resolution='l')

    rho = self.get('Rho', fname=fname, numtype=numtype, \
                   resolution=  'l', silent=silent)

    TP13 = T9**(1./3.)
    TP23 = TP13*TP13
    TP43=TP23*TP23
    thyng = np.sqrt(T9)
    TP14 = np.sqrt(thyng)
    TP12 = TP14*TP14
    TP32 = T9*TP12
    TM13 = 1./TP13
    TM23 = 1./TP23
    TM32 = 1./TP32

    # Equation 18.75 of Kippenhahn+12 with electron screening neglected.
    enuc = 2.14e37*(1.e3*rho)*XO16conv*XO16conv*TM23
    thyng = -135.93*TM13 - 0.629*TP23 - 0.445*TP43 + 0.0103*T9*T9
    enuc = enuc*np.exp(thyng)
    enuc = enuc*rho

    # This factor can account for the heating bug if present.
    enuc *= corr_fact

    return enuc

@np.vectorize
def map_signed(x, p0, p1, s0, s1):
    '''
    This function emulates the mapping of signed variables in PPMstar.

    x: input value in code units; can be a single number or a vector
    p0, p1: mapping parameters used in the first scaling step
    s0, s1: mapping parameters used in the last scaling step (just before
            the conversion to an integer)
    '''

    thyng = (x - p0)*p1
    thang = thyng * thyng  +  1.
    thang = np.sqrt(thang)
    thyng = thyng + thang
    thyng = thyng * thyng
    thang = thyng + 1.
    thyng = thyng / thang
    y = s1 * thyng  +  s0

    return y

@np.vectorize
def inv_map_signed(y, p0, p1, s0, s1):
    '''
    This function inverts map_signed().
    '''

    if y <= s0:
        x = -np.inf
    elif y >= s0 + s1:
        x = np.inf
    else:
        def func(x):
            return y - map_signed(x, p0, p1, s0, s1)

        x = optimize.newton(func, 0.)

    return x

@np.vectorize
def map_posdef(x, p0, p1, s0, s1):
    '''
    This function emulates the mapping of positive definite variables in PPMstar.

    x: input value in code units; can be a single number or a vector
    p0, p1: mapping parameters used in the first scaling step
    s0, s1: mapping parameters used in the last scaling step (just before
            the conversion to an integer)
    '''

    thyng = (x - p0)*p1
    thang = thyng * thyng  +  1.
    thang = np.sqrt(thang)
    thyng = thyng + thang
    thyng = thyng * thyng
    thang = thyng + 1.
    thyng = (thyng - 1.) / thang
    y = s1 * thyng  +  s0

    return y

@np.vectorize
def inv_map_posdef(y, p0, p1, s0, s1):
    '''
    This function inverts map_posdef().
    '''

    if y <= s0:
        x = -np.inf
    elif y >= s0 + s1:
        x = np.inf
    else:
        def func(x):
            return y - map_posdef(x, p0, p1, s0, s1)

        x = optimize.newton(func, 0.)

    return x


def make_colourmap(colours, alphas=None):
    '''
    make a matplotlib colormap given a list of index [0-255], RGB tuple values
    (normalised 0-1) and (optionally) a list of alpha index [0-255] and alpha values, i.e.:
        
    colours = [[0, (0., 0., 0.)], [1, (1., 1., 1.)]]
    alphas = [[0, 0.], [1, 1.]]
    '''

    indices_normed = np.array([float(c[0])/255. for c in colours])
    # enforce [0-1]
    if indices_normed[-1] != 1.:
        print 'first/last colour indices:', indices_normed[-1]
        print 'correcting to 1.'
        indices_normed[-1] = 1.

    rgb = colours
    cdict = {'red': [], 'green': [], 'blue': []}
    for i in range(len(colours)):
        myrgb = rgb[i][1]
        cdict['red'].append([indices_normed[i], myrgb[0], myrgb[0]])
        cdict['green'].append([indices_normed[i], myrgb[1], myrgb[1]])
        cdict['blue'].append([indices_normed[i], myrgb[2], myrgb[2]])

    if alphas!=None:
        cdict['alpha'] = []
        indices_normed = np.array([float(a[0]) / 255. for a in alphas])
        alpha = alphas
        for i in range(len(alphas)):
            myalpha = alpha[i][1]
            cdict['alpha'].append([indices_normed[i], myalpha, myalpha])

    cmap = matplotlib.colors.LinearSegmentedColormap('ppm', cdict, N=1536)

    return cmap

def colourmap_from_str(str, segment=None):
    points = []
    for line in str.splitlines():
        parts = line.split()
        if (len(parts) == 5) and (parts[0] == 'Cnot:'):
            points.append(parts[1:])
    
    points = np.array(points, dtype=np.float)
    points = points[points[:,0].argsort()]

    if segment is not None:
        # Index of the first point with value > segment[0].
        idx0 = np.argmax(points[:, 0] > segment[0])
        if idx0 > 0:
            t = (float(segment[0]) - points[idx0 - 1, 0])/ \
                (points[idx0, 0] - points[idx0 - 1, 0])
                
            new_point = (1. - t)*points[idx0 - 1, :] + t*points[idx0, :]
            points = np.vstack([new_point, points[idx0:, :]])
            
        # Index of the first point with value > segment[1].
        idx1 = np.argmax(points[:, 0] > segment[1])
        if idx1 > 0:
            t = (float(segment[1]) - points[idx1 - 1, 0])/ \
                (points[idx1, 0] - points[idx1 - 1, 0])
            
            if t > 0.:
                new_point = (1. - t)*points[idx1 - 1, :] + t*points[idx1, :]
                points = np.vstack([points[0:idx1, :], new_point])
            else:
                points = points[0:idx1, :]

    p0 = points[0, 0]
    p1 = points[-1, 0]
    for i in range(points.shape[0]):
        points[i, 0] = (points[i, 0] - p0)/(p1 - p0)
    
    r = np.zeros((points.shape[0], 3))
    r[:, 0] = points[:, 0]
    r[:, 1] = points[:, 1]
    r[:, 2] = points[:, 1]
                 
    g = np.zeros((points.shape[0], 3))
    g[:, 0] = points[:, 0]
    g[:, 1] = points[:, 2]
    g[:, 2] = points[:, 2]
                 
    b = np.zeros((points.shape[0], 3))
    b[:, 0] = points[:, 0]
    b[:, 1] = points[:, 3]
    b[:, 2] = points[:, 3]

    cmap_points = {'red': r, 'green': g, 'blue': b}
    cmap = matplotlib.colors.LinearSegmentedColormap('my_cmap', cmap_points)
    
    return cmap

class LUT():
    def __init__(self, lutfile, p0, p1, s0, s1, posdef=False):
        '''given a LUT file from the PPMstar visualisation software and the
        colour compression variables s0, s1, p0 and p0 that were used to
        compress the values of the variable being visualised (e.g., radial
        velocity), this object contains the information needed to draw
        colourbars with matplotlib for a PPMstar volume rendering.
        
        The values of s0, s1, p0 and p1 should be stored in the file
        compression_variables.txt in the setup directory of the respective
        project.
        
        Examples:
        ---------
        import ppm
        lut = ppm.LUT('./LUTS/BW-1536-UR-3.lut', s0=5., s1=245.499,
                      p0=0., p1=1.747543E-02/8.790856E-03, posdef=False)
        cbar = lut.make_colourbar([-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1])
        cbar.set_label('$v_\mathrm{r}\,/\,1000\,km\,s^{-1}$',size=6)
        draw()
        '''

        self.colours = []
        self.alphas = []
        self.cmap = None
        self.s0 = s0
        self.s1 = s1
        self.p0 = p0
        self.p1 = p1
        # is the variable positive definite (otherwise signed):
        self.posdef = posdef
        with open(lutfile,'r') as f:
            nlines = len(f.readlines())
            f.seek(0)
            for i in range(nlines):
                spl = f.readline().split()
                index = int(spl[1])
                if spl[0][0] == 'C':
                    rgb_tuple = (float(spl[2]),
                                 float(spl[3]),
                                 float(spl[4]))
                    self.colours.append([index, rgb_tuple])
                elif spl[0][0] == 'A':
                    self.alphas.append([index, float(spl[2])])
                else:
                    raise IOError("unrecognised LUT file format")
        f.close()
        self.ncolours = len(self.colours)
        self.nalphas = len(self.alphas)

    def make_colourbar(self, ticks=[], horizontal=True, background=(0,0,0),
            scale_factor=1.):
        '''make a colourbar for a PPMstar volume rendering given the mapping
        between the [0-255] colour indices and the values to which they
        correspond. returns a matplotlib.pyplot.colorbar instance, for ease of
        editing the colorbar
        
        Parameters:
        -----------
        ticks: numpy array
            at which values of the variable you would like to have ticks on the
            colourbar
        scale_factor: float
            dividing your ticks by this number should give the real values in
            code units. so, if I give ticks in km/s, I should give
            scale_factor=1.e3

        Examples:
        ---------
        import ppm
        lut = ppm.LUT('./LUTS/BW-1536-UR-3.lut', s0=5., s1=245.499, p0=0., p1=1.747543E-02/8.790856E-03, posdef=False)
        cbar=lut.make_colourbar(np.linspace(-100,100,5),background=(0.4117647058823529,0.4117647058823529,0.4235294117647059),scale_factor=1.e3)
        cbar.set_label('$v_\mathrm{r}\,/\,km\,s^{-1}$',size=6)
        draw()
        '''

        if background != (1,1,1):
            fgcolor = (1, 1, 1)
            pl.rcParams['text.color'] = fgcolor
            pl.rcParams['xtick.color'] = fgcolor
            pl.rcParams['ytick.color'] = fgcolor
            pl.rcParams['axes.edgecolor'] = fgcolor
            pl.rcParams['axes.labelcolor'] = fgcolor
            pl.rcParams['figure.edgecolor'] = background
            pl.rcParams['figure.facecolor'] = background
            pl.rcParams['savefig.edgecolor'] = background
            pl.rcParams['savefig.facecolor'] = background
            pl.rcParams['axes.facecolor'] = background

        colours = copy.deepcopy(self.colours)
        ticks = np.array(ticks)/scale_factor

        # determine which codec to use
        self.map_values = map_posdef if self.posdef else map_signed
        self.inv_map_values = inv_map_posdef if self.posdef else inv_map_signed

        # make sure we have min and max for both colour index and values, and
        # also the locations that the ticks are going to be placed
        if ticks == []:
            minidx, maxidx = self.s0+1, self.s1-1
            minval,maxval = self.inv_map_values([minidx,maxidx],self.p0,self.p1,self.s0,self.s1)
            ticks = np.linspace(minval, maxval, 8)
        else:
            # ticks given
            minval = ticks[0]; maxval = ticks[-1]
            minidx = self.map_values(minval,self.p0,self.p1,self.s0,self.s1)
            maxidx = self.map_values(maxval,self.p0,self.p1,self.s0,self.s1)

        colour_index_ticks = [self.map_values(vt,self.p0,self.p1,self.s0,self.s1) for vt in ticks]
        colour_index_ticks = np.array(colour_index_ticks)

        if any(np.isinf(colour_index_ticks)):
            print 'ticks out of range'
            return

        print 'min/max ticks being set to:', minval, maxval
        print 'corresponding to colour indices:', minidx, maxidx
        print 'ticks being placed at:', ticks
        print 'with colour indices:', colour_index_ticks

        # OK, so now we have to make the colour map on the fly (because we are
        # having to sample the original LUT for the subsection we are
        # interested in.
        
        # This means normalising the appropriate interval to the interval
        # [0-255] and constructing new left and right edges, passing the colours (and maybe alphas)
        # to the make_colourmap function.

        # left:
        i0 = np.where(np.array([c[0] for c in colours]) <= minidx)[0][-1]
        i1 = i0 + 1
        ileft = i1
        idx0 = colours[i0][0]
        idx1 = colours[i1][0]
        r0, r1 = colours[i0][1][0], colours[i1][1][0]
        g0, g1 = colours[i0][1][1], colours[i1][1][1]
        b0, b1 = colours[i0][1][2], colours[i1][1][2]
        rl = r0 + (r1 - r0)/(idx1 - idx0) * (minidx - idx0)
        gl = g0 + (g1 - g0)/(idx1 - idx0) * (minidx - idx0)
        bl = b0 + (b1 - b0)/(idx1 - idx0) * (minidx - idx0)

        # right:
        i0 = np.where(np.array([c[0] for c in colours]) <= maxidx)[0][-1]
        i1 = i0 + 1
        iright = i1
        idx0 = colours[i0][0]
        idx1 = colours[i1][0]
        r0, r1 = colours[i0][1][0], colours[i1][1][0]
        g0, g1 = colours[i0][1][1], colours[i1][1][1]
        b0, b1 = colours[i0][1][2], colours[i1][1][2]
        rr = r0 + (r1 - r0)/(idx1 - idx0) * (maxidx - idx0)
        gr = g0 + (g1 - g0)/(idx1 - idx0) * (maxidx - idx0)
        br = b0 + (b1 - b0)/(idx1 - idx0) * (maxidx - idx0)

        print ileft, iright, minidx, maxidx

        to_splice = copy.deepcopy(colours)[ileft:iright]
        newcolours = [[minidx, (rl, gl, bl)]] + to_splice + [[maxidx, (rr, gr, br)]]

        # now normalise the indices to [0-255]
        indices = np.array([c[0] for c in newcolours])
        newindices = 255.*(indices - np.min(indices)) / indices.ptp()
        # renormalise index tick locations as well
        colour_index_ticks = 255.*(colour_index_ticks - np.min(colour_index_ticks)) / colour_index_ticks.ptp()
        print 'new colour indices:', newindices
        print 'ticks now at:', colour_index_ticks

        for i in range(len(newcolours)):
            newcolours[i][0] = newindices[i]

        self.cmap = make_colourmap(newcolours)

        x = np.linspace(0, 256, 257)
        y = x.copy()
        xx, yy = np.meshgrid(x, y)
        mat = xx.copy()

        pl.figure()
        pcol = pl.pcolor(x, y, mat.T, cmap=self.cmap)
        pl.gca().xaxis.set_visible(False)
        pl.ylabel('colour index')

        if horizontal:
            cbar = pl.colorbar(orientation='horizontal', ticks = colour_index_ticks)
            cbar.ax.set_xticklabels(ticks*scale_factor)
        else:
            cbar = pl.colorbar(ticks = colour_index_ticks)
            cbar.ax.set_yticklabels(ticks*scale_factor)
        cbar.solids.set_edgecolor('face')
        cbar.ax.tick_params(axis=u'both', which=u'both',length=0,labelsize=6)
        pl.draw()

        return cbar

def cmap_from_str(str, segment=None):
    points = []
    for line in str.splitlines():
        parts = line.split()
        if (len(parts) == 5) and (parts[0] == 'Cnot:'):
            points.append(parts[1:])
    
    points = np.array(points, dtype=np.float)
    points = points[points[:,0].argsort()]

    if segment is not None:
        # Index of the first point with value > segment[0].
        idx0 = np.argmax(points[:, 0] > segment[0])
        if idx0 > 0:
            t = (float(segment[0]) - points[idx0 - 1, 0])/ \
                (points[idx0, 0] - points[idx0 - 1, 0])
                
            new_point = (1. - t)*points[idx0 - 1, :] + t*points[idx0, :]
            points = np.vstack([new_point, points[idx0:, :]])
            
        # Index of the first point with value > segment[1].
        idx1 = np.argmax(points[:, 0] > segment[1])
        if idx1 > 0:
            t = (float(segment[1]) - points[idx1 - 1, 0])/ \
                (points[idx1, 0] - points[idx1 - 1, 0])
            
            if t > 0.:
                new_point = (1. - t)*points[idx1 - 1, :] + t*points[idx1, :]
                points = np.vstack([points[0:idx1, :], new_point])
            else:
                points = points[0:idx1, :]

    p0 = points[0, 0]
    p1 = points[-1, 0]
    for i in range(points.shape[0]):
        points[i, 0] = (points[i, 0] - p0)/(p1 - p0)
    
    r = np.zeros((points.shape[0], 3))
    r[:, 0] = points[:, 0]
    r[:, 1] = points[:, 1]
    r[:, 2] = points[:, 1]
                 
    g = np.zeros((points.shape[0], 3))
    g[:, 0] = points[:, 0]
    g[:, 1] = points[:, 2]
    g[:, 2] = points[:, 2]
                 
    b = np.zeros((points.shape[0], 3))
    b[:, 0] = points[:, 0]
    b[:, 1] = points[:, 3]
    b[:, 2] = points[:, 3]

    cmap_points = {'red': r, 'green': g, 'blue': b}
    cmap = LinearSegmentedColormap('my_cmap', cmap_points)
    
    return cmap

###########################################################
# Additional plotting methods Jericho
###########################################################

def analyse_dump(rp, r1, r2):

    '''
    This function analyses ray profiles of one dump and returns

    r, ut, dutdr, r_ub,

    Parameters
    ----------
    rp: radial profile object
        radial profile
    r1: float
        minimum radius for the search for r_ub
    r2: float
        maximum radius for the search for r_ub\
        
    Output
    ------
    r: array
        radius
    ut: array
        RMS tangential velocity profiles for all buckets (except the 0th)
    dutdr: array
        radial gradient of ut for all buckets (except the 0th)
    r_ub: array
        radius of the upper boundary as defined by the minimum in dutdr
        for all buckets  (except the 0th).
        
    '''
    n_buckets = rp.get('nbuckets')

    r = rp.get_table('y')
    dr = 0.5*(np.roll(r, -1) - np.roll(r, +1))

    idx1 = np.argmin(np.abs(r - r1))
    idx2 = np.argmin(np.abs(r - r2))

    ekt = rp.get_table('ekt')
    ut = ekt[0, :, 1:n_buckets+1]**0.5

    dut = 0.5*(np.roll(ut, -1, axis = 0) - np.roll(ut, +1, axis = 0))
    dutdr = np.transpose(np.array([dut[:, i]/dr for i in range(n_buckets)]))

    idx_min_dutdr = [idx1 + np.argmin(dutdr[idx1:idx2 + 1, i]) \
                     for i in range(n_buckets)]
    r_ub = np.zeros(n_buckets)

    for bucket in range(n_buckets):
        idx = idx_min_dutdr[bucket]
        r_min = r[idx] # 0th-order estimate

        # try to fit a parabola around r_min
        r_fit = r[idx-1:idx+2]
        dutdr_fit = dutdr[idx-1:idx+2, bucket]
        coefs = np.polyfit(r_fit, dutdr_fit, 2)

        # hopefully we can determine the position of the minimum from the fit
        if coefs[0] != 0:
            r_min = -coefs[1]/(2.*coefs[0])
            # go back to 0th order if something has gone awry with the fit
            if r_min < r[idx -1] or r_min > r[idx + 1]:
                r_min = r[idx]

        r_ub[bucket] = r_min

    return r, ut, dutdr, r_ub

def upper_bound_ut(data_path, dump_to_plot, hist_dump_min, hist_dump_max, derivative = False, r1=7.4, r2=8.4):

    '''
    Finds the upper convective boundary as defined by the steepest decline in 
    tangential velocity. 

    Subpolot(1) plots the tangential velocity as a function of radius for a single dump and 
        displays the convective boundary
    Subplot(2) plots a histogram of the convective boundaries for a range of dumps specified by
        user and compares them to the selected dump

    Plots Fig. 14 or 15 in paper: "Idealized hydrodynamic simulations
    of turbulent oxygen-burning shell convection in 4 geometry"
    by Jones, S.; Andrassy, R.; Sandalski, S.; Davis, A.; Woodward, P.; Herwig, F.
    NASA ADS: http://adsabs.harvard.edu/abs/2017MNRAS.465.2991J

    Parameters
    ----------
    derivative = boolean
        True = plot dut/dr False = plot ut
    dump_To_plot = int
        The file number of the dump you wish to plot
    hist_dump_min/hist_dump_max = int
        Range of file numbers you want to use in the histogram
    r1/r2 = float
        This function will only search for the convective 
        boundary in the range between r1/r2

    Example
    -------
    data_path = "/data/ppm_rpod2/RProfiles/O-shell-M25/D15/"
    dump_to_plot = 121
    hist_dump_min = 101; hist_dump_max = 135
    r1 = 7.4; r2 = 8.4
    upper_bound_ut(data_path, derivative, dump_to_plot,\
        hist_dump_min, hist_dump_max, r1, r2)

    '''
    cb = utils.colourblind
    rp_set = rprof.rprofile_set(data_path)
    rp = rp_set.get_dump(dump_to_plot)

    n_dumps = len(rp_set.dumps)
    n_buckets = rp_set.get_dump(rp_set.dumps[0]).get('nbuckets')
    t = np.zeros(n_dumps)
    r_ub = np.zeros((n_buckets, n_dumps))

    for k in range(n_dumps):
        rp = rp_set.get_dump(rp_set.dumps[k])
        t[k] = rp.get('time')

        res = analyse_dump(rp, r1, r2)
        r = res[0]
        ut = res[1]
        dutdr = res[2]
        r_ub[:, k] = res[3]

    avg_r_ub = np.sum(r_ub, axis = 0)/float(n_buckets)
    dev = np.array([r_ub[i, :] - avg_r_ub for i in range(n_buckets)])
    sigmap_r_ub = np.zeros(n_dumps)
    sigmam_r_ub = np.zeros(n_dumps)

    for k in range(n_dumps):
        devp = dev[:, k]
        devp = devp[devp >= 0]
        if len(devp) > 0:
            sigmap_r_ub[k] = (sum(devp**2)/float(len(devp)))**0.5
        else:
            sigmam_r_ub[k] = None

        devm = dev[:, k]
        devm = devm[devm <= 0]
        if len(devm) > 0:
            sigmam_r_ub[k] = (sum(devm**2)/float(len(devm)))**0.5
        else:
            sigmam_r_ub[k] = None


    hist_bins = 0.5*(r + np.roll(r, -1))
    hist_bins[-1] = hist_bins[-2] + (hist_bins[-2] - hist_bins[-3])
    #hist_bins = np.insert(hist_bins, 0., 0.)       # #robert - this command throws an error?!?

    print "Dump {:d} (t = {:.2f} min).".format(dump_to_plot, t[dump_to_plot]/60.)
    print "Histogram constructed using dumps {:d} (t = {:.2f} min) to {:d} (t = {:.2f} min) inclusive."\
        .format(hist_dump_min, t[hist_dump_min]/60., hist_dump_max, t[hist_dump_max]/60.)

    fig = pl.figure( figsize = (2*3.39, 2*2.8))   
    #fig = pl.figure( figsize = (2*5, 2*4)) 
    gs = gridspec.GridSpec(2, 1, height_ratios = [3, 1])
    ax0 = pl.subplot(gs[0])

    if derivative:
        temp = dutdr
        lims = (-0.49, 0.1)
    else:
        temp = 1e3*ut
        lims = (-9.99, 70)

    ax0.set_ylim(lims)

    for bucket in range(n_buckets):
        lbl = r'bucket data' if bucket == 0 else None

        ax0.plot(r, temp[:, bucket], ls = '-', lw = 0.5, color = cb(3), \
            label = lbl)

        lines = (min(lims) + (max(lims)- min(lims))/13.3 ,\
                 min(lims) + (max(lims)- min(lims))/13.3 + (max(lims)- min(lims))/30)
        lbl = r'steepest decline'
        lbl = lbl if bucket == 0 else None
        ax0.plot((r_ub[bucket, dump_to_plot], r_ub[bucket, dump_to_plot]), lines, \
                 ls = '-', lw = 0.5, color = cb(4), label = lbl)

    ax0.axvline(x = avg_r_ub[dump_to_plot], ls = '--', lw = 1., color = cb(4), label = 'average')
    ax0.axvline(x = avg_r_ub[dump_to_plot] - 2*sigmam_r_ub[dump_to_plot], ls = ':', lw = 1., \
                color = cb(4), label = '2$\sigma$ fluctuations')
    ax0.axvline(x = avg_r_ub[dump_to_plot] + 2*sigmap_r_ub[dump_to_plot], ls = ':', lw = 1., color = cb(4))
    ax0.set_xlim((r1 - 0.4, r2))

    ax0.set_ylabel(r'v$_{\!\perp}$ / km s$^{-1}$')
    yticks = ax0.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    ax0.legend(loc = 3, frameon = False)

    #ax0.autoscale(enable=True, axis='y', tight=True)
    ax1 = pl.subplot(gs[1])
    ax1.hist(r_ub[:, hist_dump_min:hist_dump_max+1].flatten(), bins = hist_bins, \
             log = True, color = cb(3), edgecolor = cb(4), lw = 0.5)
    ax1.axvline(x = avg_r_ub[dump_to_plot], ls = '--', lw = 1., color = cb(4))
    ax1.axvline(x = avg_r_ub[dump_to_plot] - 2*sigmam_r_ub[dump_to_plot], ls = ':', lw = 1., color = cb(4))
    ax1.axvline(x = avg_r_ub[dump_to_plot] + 2*sigmap_r_ub[dump_to_plot], ls = ':', lw = 1., color = cb(4))
    ax1.set_xlim((r1 - 0.4, r2))
    #ax1.set_ylim((4e-1, 4e3))
    ax1.set_xlabel(r'r / Mm')
    ax1.set_ylabel(r'N')
    ax1.minorticks_off()
    fig.subplots_adjust(hspace = 0)
    pl.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible = False)
    
def get_vr_max_evolution(prof, cycles, r1, r2):

    r = prof.get('Y', fname = cycles[0], resolution = 'l')
    idx1 = np.argmin(np.abs(r - r1))
    idx2 = np.argmin(np.abs(r - r2))

    t = np.zeros(len(cycles))
    vr_max = np.zeros(len(cycles))
    for k in range(len(cycles)):
        t[k] = prof.get('t', fname = cycles[k], resolution = 'l')[-1]
        vr_rms  = prof.get('EkY', fname = cycles[k], resolution = 'l')**0.5
        vr_max[k] = np.max(vr_rms[idx2:idx1])

    return t, vr_max

def vr_evolution(cases, ymin = 4., ymax = 8.,ifig = 10):

    '''
    Compares the time evolution of the max RMS velocity of different runs

    Plots Fig. 12 in paper: "Idealized hydrodynamic simulations
    of turbulent oxygen-burning shell convection in 4 geometry"
    by Jones, S.; Andrassy, R.; Sandalski, S.; Davis, A.; Woodward, P.; Herwig, F.
    NASA ADS: http://adsabs.harvard.edu/abs/2017MNRAS.465.2991J

    Parameters
    ----------
    cases : string array
        directory names that contain the runs you wish to compare
        assumes ppm.set_YProf_path was used, thus only the directory name
        is necessary and not the full path ex. cases = ['D1', 'D2']
    ymin, ymax : float
        Boundaries of the range to look for vr_max in

    '''
    sparse = 1
    markevery = 25
    cb = utils.colourblind
    yy = 0
    
    pl.figure(ifig)
    
    for case in cases:

        try:
            prof = yprofile(ppm_path+case)
        except ValueError:
            print "have you set the yprofile filepath using ppm.set_YProf_path?"

        cycles = range(prof.cycles[0], prof.cycles[-1], sparse)
        t, vr_max = get_vr_max_evolution(prof, cycles, ymin, ymax)
        pl.plot(t/60.,  1e3*vr_max,  color = cb(yy),\
                 marker = 's', markevery = markevery, label = case)
        yy += 1

    pl.title('')
    pl.xlabel('t / min')
    pl.ylabel(r'v$_r$ / km s$^{-1}$')
    pl.legend(loc = 0)
   
def luminosity_for_dump(path, get_t = False):
    
    '''
    
    Takes a directory and returns luminosity and time vector with entry
    corresponding to each file in the dump
    
    !Can take both rprofile and yprofile filepaths
    
    Parameters
    ----------
    path : string
        Filepath for the dumps, rprofile or yprofile dumps.
    get_t: Boolean
        Return the time vector or not 

    Returns
    -------
    t : 1*ndumps array
        time vector [t/min]
    L_H : 1*ndumps array
        luminosity vector [L/Lsun]
    
    '''
    
    yprof = yprofile(path)
    
    try:
        dumps = np.arange(min(yprof.ndumpDict.keys()),\
                          #min(yprof.ndumpDict.keys())+100,1)
                          max(yprof.ndumpDict.keys()) + 1, 1)
        is_yprofile = True
    except:
        
        rp_set = rprof.rprofile_set(path)
        dumps = range(rp_set.dumps[0],\
                      #rp_set.dumps[0]+100,1)
                      rp_set.dumps[-1]+1,1)
        r_rp = rp_set.get_dump(dumps[0]).get('y')
        dV_rp = 4.*np.pi*r_rp**2*cdiff(r_rp)
        is_yprofile = False
        yprof = yprofile(path.replace('RProfiles','YProfiles'))

    airmu = 1.39165
    cldmu = 0.725
    fkair = 0.203606102635
    fkcld = 0.885906040268
    AtomicNoair = 6.65742024965
    AtomicNocld = 1.34228187919

    patience0 = 5
    patience = 10

    nd = len(dumps)
    t = np.zeros(nd)
        
    L_H = np.zeros(nd)
    
    t00 = time.time()
    t0 = t00
    k = 0
    for i in range(nd):
        if is_yprofile:
            if get_t:
                t[i] = yprof.get('t', fname = dumps[i], resolution = 'l')[-1]

            L_H[i] = yprof.get('L_C12pg', fname = dumps[i], resolution = 'l', airmu = airmu, \
                                  cldmu = cldmu, fkair = fkair, fkcld = fkcld, AtomicNoair = AtomicNoair, \
                                  AtomicNocld = AtomicNocld, corr_fact = 1.5)
        
        else:
            
            rp = rp_set.get_dump(dumps[i])
            enuc_rp = rp.get_table('enuc')
            if get_t:
                t[i] = rp.get('time')  
            # It looks like we do not need to make a correction for the heating bug here. Strange!!!
            L_H[i] = np.sum(enuc_rp[0, :, 0]*dV_rp)

        t_now = time.time()
        if (t_now - t0 >= patience) or \
           ((t_now - t00 < patience) and (t_now - t00 >= patience0) and (k == 0)):
            time_per_dump = (t_now - t00)/float(i + 1)
            time_remaining = (nd - i - 1)*time_per_dump
            print 'Processing will be done in {:.0f} s.'.format(time_remaining)
            t0 = t_now
            k += 1
    if get_t:        
        return t, L_H
    else:
        return L_H
    
def plot_luminosity(L_H_yp,L_H_rp,t):
    
    '''
    
    Plots two luminosity vectors against the same time vector
        
    Parameters
    ----------
    L_H_yp : 1 *ndumps vector
             Luminosity vector for yprofile can be generated by luminosity_for_dump
    L_H_rp : 1 *ndumps vector
             Luminosity vector for rprofile can be generated by luminosity_for_dump
    t : array size(L_H_rp)
             time vector to be plotted on the x-axis
    '''
    cb = utils.colourblind
    
    L_He = 2.25*2.98384E-03
 
    ifig = 1; pl.close(ifig); pl.figure(ifig)
    pl.semilogy(t/60., (1e43/ast.lsun_erg_s)*L_H_yp, color = cb(6), \
                 zorder = 2, label = r'L$_\mathrm{H}$')
    pl.axhline((1e43/ast.lsun_erg_s)*L_He, ls = '--', color = cb(4), \
                zorder = 1, label = r'L$_\mathrm{He}$')
    pl.xlabel('t / min')
    pl.ylabel(r'L / L$_\odot$')
    #pl.xlim((0., 2.8e3))
    #pl.ylim((1e5, 1e10))
    pl.legend(loc = 0)
    pl.tight_layout()

    ifig = 2; pl.close(ifig); pl.figure(ifig)
    pl.semilogy(t/60., (1e43/ast.lsun_erg_s)*L_H_yp, color = cb(5), \
                 lw = 2., zorder = 2, label = r'L$_\mathrm{H,yp}$')
    pl.semilogy(t/60., (1e43/ast.lsun_erg_s)*L_H_rp, color = cb(6), \
                 zorder = 4, label = r'L$_\mathrm{H,rp}$')
    pl.axhline((1e43/ast.lsun_erg_s)*L_He, ls = '--', color = cb(4), \
                zorder = 1, label = r'L$_\mathrm{He}$')
    pl.xlabel('t / min')
    pl.ylabel(r'L / L$_\odot$')
    #pl.xlim((0., 2.8e3))
    #pl.ylim((1e5, 1e10))
    pl.legend(loc = 0)
    pl.tight_layout()

def L_H_L_He_comparison(cases, ifig=101):
    
    yprofs = {}
    res = {}
    
    for case in cases:
        
        try:
            yprofs[case] = yprofile(ppm_path+case)
        except ValueError:
            print "have you set the yprofile filepath using ppm.set_YProf_path?"
        
        r = yprofs[case].get('Y', fname=0, resolution='l')
        res[case] = 2*len(r)
        
    airmu = 1.39165
    cldmu = 0.725
    fkair = 0.203606102635
    fkcld = 0.885906040268
    AtomicNoair = 6.65742024965
    AtomicNocld = 1.34228187919

    cb = utils.colourblind

    patience0 = 5
    patience = 60

    sparse = 1
    dumps = {}
    nd = {}
    t = {}
    L_H = {}
    L_He = 2.25*2.98384E-03

    for this_case in cases:
        print 'Processing {:s}...'.format(this_case)

        dumps[this_case] = np.arange(min(yprofs[case].ndumpDict.keys()),\
           max(yprofs[case].ndumpDict.keys()) + 1, sparse)
        #dumps[this_case] = np.arange(min(yprofs[case].ndumpDict.keys()),\
        #   min(yprofs[case].ndumpDict.keys()) + 10, sparse)
        #n_dumps = len(rp_set.dumps)
        nd[this_case] = len(dumps[this_case])
        t[this_case] = np.zeros(nd[this_case])
        L_H[this_case] = np.zeros(nd[this_case])

        t00 = time.time()
        t0 = t00
        k = 0
        for i in range(nd[this_case]):
            t[this_case][i] = yprofs[this_case].get('t', fname = dumps[this_case][i], \
                              resolution = 'l')[-1]
            L_H[this_case][i] = yprofs[this_case].get('L_C12pg', fname = dumps[this_case][i], \
                                resolution = 'l', airmu = airmu, cldmu = cldmu, \
                                fkair = fkair, fkcld = fkcld,  AtomicNoair = AtomicNoair, \
                                AtomicNocld = AtomicNocld, corr_fact = 1.5)

            t_now = time.time()
            if (t_now - t0 >= patience) or \
               ((t_now - t00 < patience) and (t_now - t00 >= patience0) and (k == 0)):
                time_per_dump = (t_now - t00)/float(i + 1)
                time_remaining = (nd[this_case] - i - 1)*time_per_dump
                print 'Processing will be done in {:.0f} s.'.format(time_remaining)
                t0 = t_now
                k += 1

    pl.close(ifig); pl.figure(ifig)
    pl.axhline((1e43/ast.lsun_erg_s)*L_He, ls = '--', color = cb(4), \
        label = r'L$_\mathrm{He}$')
    
    for this_case in cases:
        lbl = r'{:s} $\left({:d}^3\right)$'.format(this_case, res[this_case])
        pl.semilogy(t[this_case]/60., (1e43/ast.lsun_erg_s)*L_H[this_case], \
            ls = '-', color = cb(cases.index(this_case)), marker= 's', markevery=250/sparse, \
            label = case)
        
    pl.xlabel('t / min')
    pl.ylabel(r'L$_H$ / L$_\odot$')
    pl.legend(loc=0, ncol=2)
    pl.tight_layout()
    pl.savefig('L_H-L_He_F4_F5_F13.pdf')
    
def get_upper_bound(data_path, dump_to_plot, r1, r2):

    '''
    Returns information about the upper convective boundary
    
    Parameters
    ----------
    r1/r2 = float
        This function will only search for the convective 
        boundary in the range between r1/r2
    
    Output
    ------
    [all arrays]
    avg_r_ub : average radius of upper boundary
    sigmam_r_ub/sigmap_r_ub: 2 \sigma fluctuations in upper boundary
    r_ub : upper boundary
    t: time
    
    '''

    rp_set = rprof.rprofile_set(data_path)
    rp = rp_set.get_dump(dump_to_plot)

    n_dumps = len(rp_set.dumps)
    n_buckets = rp_set.get_dump(rp_set.dumps[0]).get('nbuckets')
    t = np.zeros(n_dumps)
    r_ub = np.zeros((n_buckets, n_dumps))

    for k in range(n_dumps):
        rp = rp_set.get_dump(rp_set.dumps[k])
        t[k] = rp.get('time')

        res = analyse_dump(rp, r1, r2)
        r = res[0]
        ut = res[1]
        dutdr = res[2]
        r_ub[:, k] = res[3]

    avg_r_ub = np.sum(r_ub, axis = 0)/float(n_buckets)
    dev = np.array([r_ub[i, :] - avg_r_ub for i in range(n_buckets)])
    sigmap_r_ub = np.zeros(n_dumps)
    sigmam_r_ub = np.zeros(n_dumps)

    for k in range(n_dumps):
        devp = dev[:, k]
        devp = devp[devp >= 0]
        if len(devp) > 0:
            sigmap_r_ub[k] = (sum(devp**2)/float(len(devp)))**0.5
        else:
            sigmam_r_ub[k] = None

        devm = dev[:, k]
        devm = devm[devm <= 0]
        if len(devm) > 0:
            sigmam_r_ub[k] = (sum(devm**2)/float(len(devm)))**0.5
        else:
            sigmam_r_ub[k] = None
            
    return(avg_r_ub, sigmam_r_ub, sigmap_r_ub, r_ub, t)
    
def plot_boundary_evolution(data_path, dump_to_plot, show_fits = False, r1=7.4, r2=8.4, t_fit_start=700):
    
    '''

    Displays the time evolution of the convective boundary

    Plots Fig. 14 or 15 in paper: "Idealized hydrodynamic simulations
    of turbulent oxygen-burning shell convection in 4 geometry"
    by Jones, S.; Andrassy, R.; Sandalski, S.; Davis, A.; Woodward, P.; Herwig, F.
    NASA ADS: http://adsabs.harvard.edu/abs/2017MNRAS.465.2991J

    Parameters
    ----------
    data_path = string
        data path
    show_fits = boolean
        show the fits used in finding the upper boundary
    r1/r2 = float
        This function will only search for the convective 
        boundary in the range between r1/r2
    t_fit_start = int
        The time to start the fit for upper boundary fit takes 
        range t[t_fit_start:-1] and computes average boundary
    '''
    cb = utils.colourblind
    
    rp_set = rprof.rprofile_set(data_path)
    rp = rp_set.get_dump(dump_to_plot)

    n_dumps = len(rp_set.dumps)
    n_buckets = rp_set.get_dump(rp_set.dumps[0]).get('nbuckets')
    t = np.zeros(n_dumps)
    r_ub = np.zeros((n_buckets, n_dumps))
    avg_r_ub, sigmam_r_ub, sigmap_r_ub, r_ub, t = get_upper_bound(data_path, dump_to_plot, r1, r2)
    
    idx_fit_start = np.argmin(np.abs(t - t_fit_start))
    t_fit_start = t[idx_fit_start]

    # fc = fit coefficients
    fc_avg = np.polyfit(t[idx_fit_start:-1], avg_r_ub[idx_fit_start:-1], 1)
    avg_fit = fc_avg[0]*t + fc_avg[1]
    fc_plus = np.polyfit(t[idx_fit_start:-1], 2.*sigmap_r_ub[idx_fit_start:-1], 1)
    plus_fit = fc_plus[0]*t + fc_plus[1]
    fc_minus = np.polyfit(t[idx_fit_start:-1], 2.*sigmam_r_ub[idx_fit_start:-1], 1)
    minus_fit = fc_minus[0]*t + fc_minus[1]

    ifig = 5; pl.close(ifig); fig = pl.figure(ifig)#, figsize = (6.0, 4.7))
    for bucket in range(n_buckets):
        lbl = 'bucket data' if bucket == 0 else None
        pl.plot(t/60., r_ub[bucket, :], ls = '-', lw = 0.5, color = cb(3), \
                 label = lbl)
    pl.plot(t/60., avg_r_ub, ls = '-', lw = 1., color = cb(4),\
             label = 'mean')
    pl.plot(t/60., avg_r_ub + 2*sigmap_r_ub, ls = '--', lw = 1., \
             color = cb(4), label = r'2$\sigma$ fluctuations')
    pl.plot(t/60., avg_r_ub - 2*sigmam_r_ub, ls = '--', lw = 1., \
             color = cb(4))
    if show_fits:
        pl.plot(t/60., avg_fit, ls = '-', lw = 0.5, color = cb(4), \
                label = r'$\mathregular{linear\ fits}$')
        pl.plot(t/60., avg_fit + plus_fit, ls = '-', lw = 0.5, color = cb(4))
        pl.plot(t/60., avg_fit - minus_fit, ls = '-', lw = 0.5, color = cb(4))
    pl.xlim((0., np.max(t)/60.))
    #plt.ylim((7.4, 8.6))
    pl.xlabel('t / min')
    pl.ylabel(r'r$_\mathrm{ub}$ / Mm')
    pl.legend(loc = 0, frameon = False)
    #fig.tight_layout()

    print 'The fitting starts at t = {:.1f} s = {:.1f} min.'.format(t_fit_start, t_fit_start/60.)
    print ''
    print 'Average:'
    print '{:.3e} Mm + ({:.3e} Mm/s)*t'.format(fc_avg[1], fc_avg[0])
    print ''
    print 'Positive fluctuations:'
    print '{:.3e} Mm + ({:.3e} Mm/s)*t'.format(fc_plus[1], fc_plus[0])
    print ''
    print 'Negative fluctuations:'
    print '{:.3e} Mm + ({:.3e} Mm/s)*t'.format(fc_minus[1], fc_minus[0])
    
def prof_time(profile, fname,yaxis_thing='vY',num_type='ndump',logy=False,
              radbase=None,radtop=None,ifig=101,ls_offset=0,label_case="",metric = 0,
              **kwargs):
    """
    Plot the time evolution of a profile from multiple
    dumps of the same run (...on the same figure).

    Velocities 'v', 'vY' and/or 'vXZ' may also be plotted.

    Parameters
    ----------

    fname : int or list
        Cycle or list of cycles to plot. Could also be time
        in minutes (see num_type).
    yaxis_thing : string
        What should be plotted on the y-axis?
        In addition to the dcols quantities, you may also choose
        'v', 'vY', and 'vXZ' -- the RMS 'total', radial and
        tangential velocities.
        The default is 'vY'.
    logy : boolean
        Should the velocity axis be logarithmic?
        The default value is False
    radbase, radtop : float
        Radii of the base and top of the convective region,
        respectively. If not None, dashed vertical lines will
        be drawn to mark these boundaries.
        The default value is None.
    num_type : string, optional
        Designates how this function acts and how it interprets
        fname.  If numType is 'file', this function will get the
        desired attribute from that file.  If numType is 'NDump'
        function will look at the cycle with that nDump.  If
        numType is 'T' or 'time' function will find the _cycle
        with the closest time stamp.  The default is "ndump".
    ifig : integer
        figure number
    ls_offset : integer
        linestyle offset for argument in utils.linestyle for plotting
        more than one case
    label_case : string
        optional extra label for case

    Examples
    --------

        import ppm
        run='/rpod3/fherwig/PPM/RUNS_DATA/VLTP_MESA_M0.542/C1'
        YY=ppm.yprofile(run)
        YY.prof_time([0,5,10,15,20,25],logy=False,num_type='time',radbase=10.7681,radtop=23.4042)

    """

    #fsize=14

    #params = {'axes.labelsize':  fsize,
    #    'font.family':       'serif',
    #'font.family':        'Times New Roman',
    #'figure.facecolor':  'white',
    #'text.fontsize':     fsize,
    #'legend.fontsize':   fsize,
    #'xtick.labelsize':   fsize*0.8,
    #'ytick.labelsize':   fsize*0.8,
    #'text.usetex':       False}
    #pl.rcParams.update(params)

    if type(fname) is not list:
        fname = [fname]
    if num_type is 'time':
            fname = [f * 60 for f in fname]

    pl.figure(ifig)
    i=0
    for dump in fname:
        # FH: I am changing this. The double resolution data
        # in YProfiles is broken and needs to be reduced.
        # if yaxis_thing in ['j','Y','FVconv','UYconv','FV H+He',\
        #                    'UY H+He','Rho','Rho1','A']:
        #     Y=self.get('Y',fname=dump)

        # else:
        Y=profile.get('Y',fname=dump,resolution='L')

        if yaxis_thing is 'v':
            Ek = profile.get('Ek',fname=dump,numtype=num_type,resolution='l')
            v = np.sqrt(2.*array(Ek,dtype=float))
            y = v*1000
            if logy:
                ylab = '$\log <u>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'
            else:
                ylab = '$<u>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'        
        elif yaxis_thing is 'vY':

            EkY  = profile.get('EkY',fname=dump,numtype=num_type,resolution='l', metric = metric)

            vY = np.sqrt(array(EkY,dtype=float))  # no factor 2 for v_Y and v_XZ
            y = vY*1000
            if logy:
                ylab = '$\log <u_r>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'
            else:
                ylab = '$<u_r>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'
        elif yaxis_thing is 'vXZ':
            EkXZ = profile.get('EkXZ',fname=dump,numtype=num_type,resolution='l')
            vXZ = np.sqrt(array(EkXZ,dtype=float))  # no factor 2 for v_Y and v_XZ
            y = vXZ*1000
            if logy:
                ylab = '$\log <u_{\\theta,\phi}>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'
            else:
                ylab = '$<u_{\\theta,\phi}>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'
        else:
            y = profile.get(yaxis_thing,fname=dump,numtype=num_type,resolution='L', **kwargs)
            ylab = yaxis_thing
            if logy: ylab = 'log '+ylab
        if num_type is 'ndump':
            lab = label_case+', '+str(dump)
            leg_tit = num_type
        elif num_type is 'time':
            #idx = np.abs(profile.get('t')-dump).argmin()
            time = profile.get('t', fname=dump, numtype=num_type)#[idx]
            time_min = time/60.
            lab=label_case+', '+str("%.3f" % time_min)
            leg_tit = 'time / min'

        if logy:
            pl.plot(Y,np.log10(y),utils.linestyle(i+ls_offset)[0],
                    markevery=utils.linestyle(i+ls_offset)[1],label=lab)
        else:
            pl.plot(Y,y,utils.linestyle(i+ls_offset)[0],
                    markevery=utils.linestyle(i+ls_offset)[1],label=lab)

        if radbase is not None and dump is fname[0]:
            pl.axvline(radbase,linestyle='dashed',color='k')
        if radtop is not None and dump is fname[0]:
            pl.axvline(radtop,linestyle='dashed',color='k')

        i+=1

    pl.xlabel('Radius $[1000\mathrm{km}]$')
    pl.ylabel(ylab)
    pl.legend(loc='best',title=leg_tit).draw_frame(False)