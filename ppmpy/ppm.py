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
from numpy import *
import numpy as np
from math import *
from nugridpy.data_plot import *
from nugridpy import utils
import matplotlib.pylab as pyl
import matplotlib.pyplot as pl
from matplotlib import rcParams
from matplotlib import gridspec
import nugridpy.mesa as ms
import os
import re
import nugridpy.astronomy as ast
import scipy.interpolate as interpolate
from scipy import optimize
import copy
import sys
sys.path.insert(0, '/data/ppm_rpod2/lib/lcse')
#import rprofile as rprof

cb = utils.colourblind

# from rprofile import rprofile_reader

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


class yprofile(DataPlot):
    """ 
    Data structure for holding data in the  YProfile.bobaaa files.
    
    Parameters
    ----------
    sldir : string
        which directory we are working in.  The default is '.'.
        
    """

    def __init__(self, sldir='.', filename_offset=0, silent = True):
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
                print('ppm_path not correctly set: '+sldir+' is not directory.')
        self.sldir = sldir
        if not os.path.isdir(sldir):  # If the path still does not exist
            print('error: Directory, '+sldir+ ' not found')
            print('Now returning None')
            return None
        else:
            f=os.listdir(sldir) # reads the directory
            for i in range(len(f)):  # Removes any files that are not YProfile files
                if re.search(r"^YProfile-01-[0-9]{4,4}.bobaaa$",f[i]):
                    self.files.append(f[i])
            self.files.sort()
            if len(self.files)==0: # If there are no YProfile files in the directory
                print('Error: no YProfile named files exist in Directory')
                print('Now returning None')
                return None
            slname=self.files[len(self.files)-1] #
            self.slname = slname
            if not silent:
                print("Reading attributes from file ",slname)
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
            if not silent:
                print('There are '+str(len(self.files))+' YProfile files in the ' +self.sldir+' directory.')
                print('Ndump values range from '+str(min(self.ndumpDict.keys()))+' to '+str(max(self.ndumpDict.keys())))
            t=self.get('t',max(self.ndumpDict.keys()))
            t1=self.get('t',min(self.ndumpDict.keys()))
            if not silent:
                print('Time values range from '+ str(t1[-1])+' to '+str(t[-1]))
            self.cycles=list(self.ndumpDict.keys())
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
        for i in range(len(fileList)):
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
            silent=True, **kwargs):
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
                print("Warning at yprofile.get(): fname is None, "\
                      "the last dump (%d) will be used." \
                      % max(self.ndumpDict.keys()))

        if attri in self.cattrs: # if it is a cycle attribute
            isCyc = True
        elif attri in self.dcols:#  if it is a column attribute
            isCol = True
        elif attri in self.hattrs:# if it is a header attribute
            isHead = True

        # directing to proper get method
        if isCyc:
            return self.getCycleData(attri,fname, numtype, resolution=resolution, \
                                     silent=silent)
        if isCol:
            return self.getColData(attri,fname, numtype, resolution=resolution, \
                                   silent=silent)
        if isHead:
            return self.getHeaderData(attri, silent=silent)
        else:
            res = self.computeData(attri, fname, numtype, silent=silent, **kwargs)
            if res is None:             
                if not silent:
                    print('That Data name does not appear in this YProfile Directory')
                    print('Returning none')
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
                print('The attribute '+attri+' does not appear in these YProfiles')
                print('Returning None')
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
                print("Warning at yprofile.getCycleData(): FName is None, "\
                      "the last dump (%d) will be used." % \
                      max(self.ndumpDict.keys()))

        isCyc= False #If Attri is in the Cycle Atribute section
        boo=True
        filename=self.findFile(FName, numType, silent=silent)
        data=0

        if attri in self._cycle: #if attri is a cycle attribute rather than a top attribute
            isCyc = True

        if attri not in self._cycle and isCyc:# Error checking
            if not silent:
                print('Sorry that Attribute does not appear in the fille')
                print('Returning None')
            return None

        if not Single and isCyc:

            data= self.getColData( attri,filename,'file',resolution, True)

            return data
        if Single and isCyc:

            data= self.getColData( attri,filename,'file',resolution, True)
            if data==None:
                return None
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
        #print FName
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
                print("That File, "+FName+ ", does not exist.")
                print('Returning None')
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
                print("Attribute DNE in file, Returning None")
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

    def computeData(self, attri, fname, numtype = 'ndump', silent=False,\
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

        if attri == 'T9' or attri == 'mu':
            required_args = ('airmu', 'cldmu')
            missing_args = get_missing_args(required_args, **kwargs)
            if len(missing_args) > 0:
                if not silent:
                    print('The following arguments are missing: ', \
                          missing_args)
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
            
            if attri == 'mu':
                return mu
            else:
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
                    print('The following arguments are missing: ', \
                    missing_args)
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
                    print('The following arguments are missing: ', \
                    missing_args)
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
                    print('The following arguments are missing: ', \
                    missing_args)
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

    def _get_enuc_C12pg(self, fname, airmu, cldmu, fkair, fkcld, \
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
        
    def _get_enuc_C12C12(self, fname, airmu, cldmu, fkcld, AtomicNocld,\
                        numtype='ndump', silent=False, Q=9.35, \
                        corr_fact=1., corr_func=None, T9_func=None):
        if T9_func is None:
            print('Corrected T9 profile not supplied, using uncorrected T9.')
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
        
    def _get_enuc_O16O16(self, fname, airmu, cldmu, XO16conv, \
                         numtype='ndump', silent=False, corr_fact=1., \
                         T9_func=None):
        if T9_func is None:
            print('Corrected T9 profile not supplied, using uncorrected T9.')
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
                    print('Improper value for NDump, choosing 0 instead')
                FName=0
            if FName < 0:
                if not silent:
                    print('User Cant select a negative NDump')
                    print('Reselecting NDump as 0')
                FName=0
            if FName not in list(self.ndumpDict.keys()):
                if not silent:
                    print('NDump '+str(FName)+ ' Does not exist in this directory')
                    print('Reselecting NDump as the largest in the Directory')
                    print('Which is '+ str(max(self.ndumpDict.keys())))
                FName=max(self.ndumpDict.keys())
            boo=True

        elif numType=='T' or numType=='TIME':
            try:    #ensuring FName can be a proper time, ie no letters
                FName=float(FName)
            except:
                if not silent:
                    print('Improper value for time, choosing 0 instead')
                FName=0
            if FName < 0:
                if not silent:
                    print('A negative time does not exist, choosing a time = 0 instead')
                FName=0
            timeData=self.get('t',self.ndumpDict[max(self.ndumpDict.keys())],numtype='file')
            keys=list(self.ndumpDict.keys())
            keys.sort()
            tmp=[]
            for i in range(len(keys)):
                tmp.append(timeData[keys[i]])

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
                    print('The closest time is at Ndump = ' +str(keys[indexL]))
                FName=keys[indexL]
            else:
                if not silent:
                   print('The closest time is at Ndump = ' +str(keys[indexH]))
                FName=keys[indexH]
            boo=True
        else:
            if not silent:
                print('Please enter a valid numType Identifyer')
                print('Returning None')
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
            print('Header atribute error, directory has two YProfiles that have different header sections')
            print('Returning unchanged header')
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
        #print "Analyzing headers ..."
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

# below are some plotting functions integrated from the PPMstar_svn
# server utils/YProfPy directory
    
    def prof_time(self,fname,yaxis_thing='vY',num_type='ndump',logy=False,
                  radbase=None,radtop=None,ifig=101,ls_offset=0,label_case="",markevery = None,
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
            Y=self.get('Y',fname=dump,resolution='L')
            cb = utils.colourblind
            
            if yaxis_thing is 'v':
                Ek = self.get('Ek',fname=dump,numtype=num_type,resolution='l')
                v = np.sqrt(2.*array(Ek,dtype=float))
                y = v*1000
                if logy:
                    ylab = '$\log <u>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'
                else:
                    ylab = '$<u>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'        
            elif yaxis_thing is 'vY':
                EkY  = self.get('EkY',fname=dump,numtype=num_type,resolution='l')
                vY = np.sqrt(array(EkY,dtype=float))  # no factor 2 for v_Y and v_XZ
                y = vY*1000
                if logy:
                    ylab = '$\log <u_r>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'
                else:
                    ylab = '$<u_r>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'
            elif yaxis_thing is 'vXZ':
                EkXZ = self.get('EkXZ',fname=dump,numtype=num_type,resolution='l')
                vXZ = np.sqrt(array(EkXZ,dtype=float))  # no factor 2 for v_Y and v_XZ
                y = vXZ*1000
                if logy:
                    ylab = '$\log <u_{\\theta,\phi}>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'
                else:
                    ylab = '$<u_{\\theta,\phi}>_\mathrm{rms}$ $([u]=\mathrm{km/s})$'
                 
            elif yaxis_thing is 'FV':
                y = self.get('FV H+He',fname=dump,numtype=num_type,resolution='l')
                if logy:
                    ylab = '$\log_{10}$ fractional volume'
                else:
                    ylab = 'fractional volume'
                    
            else:
                y = self.get(yaxis_thing,fname=dump,numtype=num_type,resolution='L', **kwargs)
                ylab = yaxis_thing
                if logy: ylab = 'log '+ylab

            if num_type is 'ndump':
                lab = label_case+', '+str(dump)
                leg_tit = num_type
            elif num_type is 'time':
                idx = np.abs(self.get('t')-dump).argmin()
                time = self.get('t')[idx]
                time_min = time/60.
                lab=label_case+', '+str("%.3f" % time_min)
                leg_tit = 'time / min'
            
            if markevery is not None:
                markevery = markevery
            else:
                markevery = utils.linestyle(i+ls_offset)[1]
            if logy:
                pl.plot(Y,np.log10(y),utils.linestyle(i+ls_offset)[0],
                        markevery= markevery,label=lab,
                       color = cb(i))
            else:
                pl.plot(Y,y,utils.linestyle(i+ls_offset)[0],
                        markevery=markevery,label=lab,
                       color = cb(i))

            if radbase is not None and dump is fname[0]:
                pl.axvline(radbase,linestyle='dashed',color='k')
            if radtop is not None and dump is fname[0]:
                pl.axvline(radtop,linestyle='dashed',color='k')

            i+=1
                    
        pl.xlabel('Radius $[1000\mathrm{km}]$')
        pl.ylabel(ylab)
        pl.legend(loc='best',title=leg_tit).draw_frame(False)
    
    def get_mass_fraction(fluid,fname,resolution):
        '''
            Get mass fraction profile of fluid 'fluid' at fname with resolution
            'resolution'.
        '''
        y = self.get(fluid,fname=fname,resolution=resolution)
        if fluid == 'FV H+He':
            rhofluid = self.get('Rho H+He',fname=fname,resolution=resolution)
        else:
            rhofluid = self.get('RHOconv',fname=fname,resolution=resolution)
        rho = self.get('Rho',fname=fname,resolution=resolution)
        y = rhofluid * y / rho
        return y

    def vprofs(self,fname,fname_type = 'discrete',log_logic=False,lims=None,save=False,
               prefix='PPM',format='pdf',initial_conv_boundaries=True,
               lw=1., label=True, ifig = 11, which_to_plot = [True,True,True]):
        """
        Plot velocity profiles v_tot, v_Y and v_XZ for a given cycle number
        or list of cycle numbers (fname).
        If a list of cycle number is given, separate figures are made for
        each cycle. If one wishes to compare velocity profiles for two or
        more cycles, see function vprof_time.
            
        Parameters
        ----------
        
        fname : int or list
            Cycle number or list of cycle numbers to plot
            
        fname_type : string
            'discrete' or 'range' whether to average over a range
            to find velocities or plot entries discretely
            default is 'discrete'
        log_logic : boolean
            Should the velocity axis be logarithmic?
            The default value is False
        lims : list
            Limits for the plot, i.e. [xl,xu,yl,yu].
            If None, the default values are used.
            The default is None.
        save : boolean
            Do you want the figures to be saved for each cycle?
            Figure names will be <prefix>-Vel-00000000001.<format>,
            where <prefix> and <format> are input options that default
            to 'PPM' and 'pdf'.
            The default value is False.
        prefix : string
            see 'save' above
        format : string
            see 'save' above
        initial_conv_boundaries : logical 
            plot vertical lines where the convective boundaries are
            initially, i.e. ad radbase and radtop from header
            attributes in YProfiles
        lw : float, optional
            line width of the plot
        label : string, optional
            label for the line, if multiple models plotted
        which_to_plot : boolean array, optional
            booleans as to whether to plot v_tot v_Y and v_XZ
            respectively True to plot

        Examples
        --------
            
            import ppm
            run='/rpod3/fherwig/PPM/RUNS_DATA/VLTP_MESA_M0.542/C1'
            YY=ppm.yprofile(run)
            YY.vprofs([90,100],log_logic=True)

        """
            
        ## fsize=14
        
        ## params = {'axes.labelsize':  fsize,
        ## #    'font.family':       'serif',
        ## 'font.family':        'Times New Roman',
        ## 'figure.facecolor':  'white',
        ## 'text.fontsize':     fsize,
        ## 'legend.fontsize':   fsize,
        ## 'xtick.labelsize':   fsize*0.8,
        ## 'ytick.labelsize':   fsize*0.8,
        ## 'text.usetex':       False}
        ## pl.rcParams.update(params)
        
        pl.figure(ifig)
        if type(fname) is not list:
            fname = [fname]
        Y=self.get('Y',fname=fname[0],resolution='l')
        nn = 0
        vXZtot = np.zeros(size(Y))
        vtot = np.zeros(size(Y))
        vYtot =  np.zeros(size(Y))
        
        for dump in fname:
#            if save or dump == fname[0]:
#                pl.close(ifig),pl.figure(ifig)
#            if not save and dump != fname[0]:
#                pl.figure()
            Ek   = self.get('Ek',fname=dump,resolution='l')
            EkY  = self.get('EkY',fname=dump,resolution='l')
            EkXZ = self.get('EkXZ',fname=dump,resolution='l')
            v =   np.sqrt(2.*array(Ek,dtype=float))
            vY =  np.sqrt(array(EkY,dtype=float))  # no factor 2 for v_Y and v_XZ
            vXZ = np.sqrt(array(EkXZ,dtype=float))  # no factor 2 for v_Y and v_XZ
            
            line_labels = ['$v$','$v_\mathrm{r}$','$v_\perp$']
            styles = ['-','--',':']
            cb = utils.colourblind
            if fname_type == 'discrete':
                if log_logic:
                    if label:
                        if which_to_plot[0]:
                            pl.plot(Y,np.log10(v*1000.),\
                                ls=styles[0],\
                                color=cb(0),\
                                label=line_labels[0],\
                                lw=lw)
                        if which_to_plot[1]:
                            pl.plot(Y,np.log10(vY*1000.),\
                                ls=styles[1],\
                                color=cb(8),\
                                label=line_labels[1],\
                                lw=lw)
                        if which_to_plot[2]:
                            pl.plot(Y,np.log10(vXZ*1000.),\
                                ls=styles[2],\
                                color=cb(2),\
                                label=line_labels[2],\
                                lw=lw)
                        ylab='log v$_\mathrm{rms}$ [km/s]'
                    else:
                        if which_to_plot[0]:
                            pl.plot(Y,np.log10(v*1000.),\
                                ls=styles[0],\
                                color=cb(0),\
                                lw=lw)
                        if which_to_plot[1]:
                            pl.plot(Y,np.log10(vY*1000.),\
                                ls=styles[1],\
                                color=cb(8),\
                                lw=lw)
                        if which_to_plot[2]:
                            pl.plot(Y,np.log10(vXZ*1000.),\
                                ls=styles[2],\
                                color=cb(2),\
                                lw=lw)
                        ylab='v$_\mathrm{rms}\,/\,\mathrm{km\,s}^{-1}$'
                else:
                    if label:
                        if which_to_plot[0]:
                            pl.plot(Y,v*1000.,\
                                ls=styles[0],\
                                color=cb(0),\
                                label=line_labels[0],\
                                lw=lw)
                        if which_to_plot[1]:
                            pl.plot(Y,vY*1000.,\
                                ls=styles[1],\
                                color=cb(8),\
                                label=line_labels[1],\
                                lw=lw)
                        if which_to_plot[2]:
                            pl.plot(Y,vXZ*1000.,\
                                ls=styles[2],\
                                color=cb(2),\
                                label=line_labels[2],\
                                lw=lw)
                    else:
                        if which_to_plot[0]:
                            pl.plot(Y,v*1000.,\
                                ls=styles[0],\
                                color=cb(0),\
                                lw=lw)
                        if which_to_plot[1]:
                            pl.plot(Y,vY*1000.,\
                                ls=styles[1],\
                                color=cb(8),\
                                lw=lw)
                        if which_to_plot[2]:
                            pl.plot(Y,vXZ*1000.,\
                                ls=styles[2],\
                                color=cb(2),\
                                lw=lw)
                    ylab='v$_\mathrm{rms}\,/\,\mathrm{km\,s}^{-1}$'
                if initial_conv_boundaries:
                    pl.axvline(self.radbase,linestyle='dashed',color='k')
                    pl.axvline(self.radtop,linestyle='dashed',color='k')
                if lims is not None:
                    pl.axis(lims)
                pl.xlabel('r / Mm')
                pl.ylabel(ylab)
                pl.title(prefix+', Dump '+str(dump))
                if label:
                    pl.legend(loc=8).draw_frame(False)
                number_str=str(dump).zfill(11)
                if save:
                    pl.savefig(prefix+'-Vel-'+number_str+'.'+format,format=format)
            else:             
                vtot = vtot + v
                vYtot = vYtot + vY
                vXZtot = vXZtot + vXZ
                nn += 1
        if fname_type == 'range':
            v = vtot/nn
            vY = vYtot/nn
            #vXZ = VXZtot/nn
            if log_logic:
                    if label:
                        if which_to_plot[0]:
                            pl.plot(Y,np.log10(v*1000.),\
                                ls=styles[0],\
                                color=cb(0),\
                                label=line_labels[0],\
                                lw=lw)
                        if which_to_plot[1]:
                            pl.plot(Y,np.log10(vY*1000.),\
                                ls=styles[1],\
                                color=cb(8),\
                                label=line_labels[1],\
                                lw=lw)
                        if which_to_plot[2]:
                            pl.plot(Y,np.log10(vXZ*1000.),\
                                ls=styles[2],\
                                color=cb(2),\
                                label=line_labels[2],\
                                lw=lw)
                        ylab='log v$_\mathrm{rms}$ [km/s]'
                    else:
                        if which_to_plot[0]:
                            pl.plot(Y,np.log10(v*1000.),\
                                ls=styles[0],\
                                color=cb(0),\
                                lw=lw)
                        if which_to_plot[1]:
                            pl.plot(Y,np.log10(vY*1000.),\
                                ls=styles[1],\
                                color=cb(8),\
                                lw=lw)
                        if which_to_plot[2]:
                            pl.plot(Y,np.log10(vXZ*1000.),\
                                ls=styles[2],\
                                color=cb(2),\
                                lw=lw)
                        ylab='v$_\mathrm{rms}\,/\,\mathrm{km\,s}^{-1}$'
            else:
                    if label:
                        if which_to_plot[0]:
                            pl.plot(Y,v*1000.,\
                                ls=styles[0],\
                                color=cb(0),\
                                label=line_labels[0],\
                                lw=lw)
                        if which_to_plot[1]:
                            pl.plot(Y,vY*1000.,\
                                ls=styles[1],\
                                color=cb(8),\
                                label=line_labels[1],\
                                lw=lw)
                        if which_to_plot[2]:
                            pl.plot(Y,vXZ*1000.,\
                                ls=styles[2],\
                                color=cb(2),\
                                label=line_labels[2],\
                                lw=lw)
                    else:
                        if which_to_plot[0]:
                            pl.plot(Y,v*1000.,\
                                ls=styles[0],\
                                color=cb(0),\
                                lw=lw)
                        if which_to_plot[1]:
                            pl.plot(Y,vY*1000.,\
                                ls=styles[1],\
                                color=cb(8),\
                                lw=lw)
                        if which_to_plot[2]:
                            pl.plot(Y,vXZ*1000.,\
                                ls=styles[2],\
                                color=cb(2),\
                                lw=lw)
                    ylab='v$_\mathrm{rms}\,/\,\mathrm{km\,s}^{-1}$'
            if initial_conv_boundaries:
                    pl.axvline(self.radbase,linestyle='dashed',color='k')
                    pl.axvline(self.radtop,linestyle='dashed',color='k')
            if lims is not None:
                    pl.axis(lims)
            pl.xlabel('r / Mm')
            pl.ylabel(ylab)
            pl.title(prefix+', Dump '+str(dump))
            if label:
                    pl.legend(loc=8).draw_frame(False)
            number_str=str(dump).zfill(11)
            if save:
                    pl.savefig(prefix+'-Vel-'+number_str+'.'+format,format=format)
            
    def vprof_time(self,dumps,Np,comp = 'r',lims=None,save=False,
               prefix='PPM',format='pdf',initial_conv_boundaries=True,lw=1., ifig = 12):
        '''
        Plots the same velocity profile at different times
        
        Parameters
        ----------
        dumps : array
            dumps to plot
        Np : int
            This function averages over a range of Np points centered at 
            the dump number, preferable an even number or else the range 
            will not be centered around the dump number
        comp : string
            'r' 't' or 'tot' for the velocity component that will be plotted
            see vprof for comparing all three
        lims : list
            Limits for the plot, i.e. [xl,xu,yl,yu].
            If None, the default values are used.
            The default is None.
        save : boolean
            Do you want the figures to be saved for each cycle?
            Figure names will be <prefix>-Vel-00000000001.<format>,
            where <prefix> and <format> are input options that default
            to 'PPM' and 'pdf'.
            The default value is False.
        prefix : string
            see 'save' above
        format : string
            see 'save' above
        initial_conv_boundaries : logical 
            plot vertical lines where the convective boundaries are
            initially, i.e. ad radbase and radtop from header
            attributes in YProfiles
        ifig : int
            figure number to plot into
        '''
        
        avg_rms_v = {}
        for d in dumps:
            # Always take an average except the very end of the run.
            v = get_avg_rms_velocities(self, list(range(d-int(Np/2), d+int(Np/2))),comp)

            avg_rms_v[d] = v
        
        r = self.get('Y', fname=0, resolution='l')
        markers = ['v', '^', '<', '>', 'o', 's']
        colours = [9, 3, 5, 8, 1, 6]

        ifig = ifig; pl.close(ifig); pl.figure(ifig)
        for i in range(len(dumps)):
            pl.semilogy(r, 1e3*avg_rms_v[dumps[i]], '-', color=cb(colours[i]), \
                         marker=markers[i], markevery=50, \
                         label=r'$\tau_{{{:d}}}$'.format(i+1))
        pl.xlabel('r / Mm')
        pl.ylabel(r'v$_\mathrm{%s}$ / km s$^{-1}$' % comp)
        pl.legend(loc = 0)
        if initial_conv_boundaries:
            pl.axvline(self.radbase,linestyle='dashed',color='k')
            pl.axvline(self.radtop,linestyle='dashed',color='k')
        if lims is not None:
            pl.axis(lims)
        pl.legend(loc=0, ncol=2)
        if save:
            number_str=str(dump).zfill(11)
            pl.savefig(prefix+'-Vel-'+number_str+'.'+format,format=format)
    
    def Aprof_time(self,tau,Nl,lims=None,save=False,silent = True,
                   prefix='PPM',format='pdf',initial_conv_boundaries=True,lw=1., ifig = 12):
        '''
        Plots the same velocity profile at different times

        Parameters
        ----------
        tau : array
            times to plot
        Nl : int
            Number of lines to plot, spaced over every minute
        lims : list
            Limits for the plot, i.e. [xl,xu,yl,yu].
            If None, the default values are used.
            The default is None.
        save : boolean
            Do you want the figures to be saved for each cycle?
            Figure names will be <prefix>-Vel-00000000001.<format>,
            where <prefix> and <format> are input options that default
            to 'PPM' and 'pdf'.
            The default value is False.
        silent: boolean,optional
            Should plot display output?
        prefix : string
            see 'save' above
        format : string
            see 'save' above
        initial_conv_boundaries : logical 
            plot vertical lines where the convective boundaries are
            initially, i.e. ad radbase and radtop from header
            attributes in YProfiles
        ifig : int
            figure number to plot into
        '''

        r = self.get('Y', fname = 0, resolution = 'l')
        A0 = self.get('A', fname = 0., numtype = 'time', resolution = 'l', silent = True)
        markers = ['v', '^', '<', '>', 'o', 's']
        colours = [9, 3, 5, 8, 1, 6]

        ifig = ifig; pl.close(ifig); pl.figure(ifig)
        for i in range(Nl):
            t = self.get('t', fname = 60.*100.*i, numtype = 'time', resolution = 'l', silent = True)[-1]
            A = self.get('A', fname = 60.*100.*i, numtype = 'time', resolution = 'l', silent = True)
            pl.plot(r, A/A0 - 1., ':', lw = 0.5, color = cb(4))
        for i in range(len(tau)):
            t = self.get('t', fname = 60.*tau[i], numtype = 'time', resolution = 'l', silent = False)[-1]
            A = self.get('A', fname = 60.*tau[i], numtype = 'time', resolution = 'l', silent = True)
            pl.plot(r, A/A0 - 1., '-', marker = markers[i], color = cb(colours[i]), \
                     markevery = 50, label = r'$\tau_{{{:d}}}$'.format(i+1))
            if not silent:
                print('You wanted tau = {:.1f} min. yprofile.get() found the closest dump at t = {:.1f} min.\n'.\
                  format(tau[i], t/60.))
        pl.xlabel('r / Mm')
        pl.ylabel('A(t)/A(0) - 1')      
        if initial_conv_boundaries:
            pl.axvline(self.radbase,linestyle='dashed',color='k')
            pl.axvline(self.radtop,linestyle='dashed',color='k')
        if lims is not None:
            pl.axis(lims)
        pl.legend(loc=0)
        pl.tight_layout() 
        if save:
            number_str=str(dump).zfill(11)
            pl.savefig(prefix+'-A-'+number_str+'.'+format,format=format)
        
    def tEkmax(self,ifig=None,label=None,save=False,prefix='PPM',format='pdf',
               logy=False,id=0):
        """
        Plot maximum kinetic energy as a function of time.
        
        Parameters
        ----------
        ifig : int, optional
            Figure number. If None, chose automatically.
            The default is None.
        label : string, optional
            Label for the model
            The default is None.
        save : boolean, optional
            Do you want the figures to be saved for each cycle?
            Figure names will be <prefix>-t-EkMax.<format>,
            where <prefix> and <format> are input options that default
            to 'PPM' and 'pdf'.
            The default value is False.
        prefix : string, optional
            see 'save' above
        format : string, optional
            see 'save' above
        logy : boolean, optional
            Should the y-axis have a logarithmic scale?
            The default is False
        id : int, optional
            An id for the model, which esures that the lines are
            plotted in different colours and styles.
            The default is 0
        
        Examples
        --------
        
            import ppm
            run1='/rpod3/fherwig/PPM/RUNS_DATA/VLTP_MESA_M0.542/C1'
            run2='/rpod3/fherwig/PPM/RUNS_DATA/sakurai-num-exp-robustness-onset-GOSH/A1/'
            YY=ppm.yprofile(run1)
            YY2=ppm.yprofile(run2)
            YY.tEkmax(ifig=1,label='VLTP_0.543',id=0)
            YY2.tEkmax(ifig=1,label='Sak_A1',id=1)
        
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
        
        if ifig is None:
            pl.figure()
        else:
            pl.figure(ifig)
        
        t = self.get('t')
        EkMax = self.get('EkMax') # Ek in 10^43 erg
        
        if logy:
            y = np.log10(EkMax*1.e43)
            ylab = '$\log E_{\\rm k,max}/ {\\rm erg}$'
        else:
            y = EkMax * 1.e5 # EkMax in 10^38 erg
            ylab = '$E_{\\rm k,max} / 10^{38}{\\rm \, erg}$'
        if label is not None:
            pl.plot(t/60,y,utils.linestyle(id)[0], markevery=utils.linestyle(id)[1],
                    label=label)
        else:
            pl.plot(t/60,y,utils.linestyle(id)[0], markevery=utils.linestyle(id)[1],
                    label=str(id))
        pl.legend(loc='best').draw_frame(False)
        pl.xlabel('t/min')
        pl.ylabel(ylab)
        if save:
            pl.savefig(prefix+'-t-EkMax.'+format,format=format)

    def tvmax(self,ifig=None,label=None,save=False,prefix='PPM',format='pdf',
              logy=False,id=0):
        """
        Plot maximum velocity as a function of time.
        
        Parameters
        ----------
        ifig : int, optional
            Figure number. If None, chose automatically.
            The default is None.
        label : string, optional
            Label for the model
            The default is None.
        save : boolean
            Do you want the figures to be saved for each cycle?
            Figure names will be <prefix>-t-vMax.<format>,
            where <prefix> and <format> are input options that default
            to 'PPM' and 'pdf'.
            The default value is False.
        prefix : string
            see 'save' above
        format : string
            see 'save' above
        logy : boolean, optional
            Should the y-axis have a logarithmic scale?
            The default is False
        id : int, optional
            An id for the model, which esures that the lines are
            plotted in different colours and styles.
            The default is 0

        Examples
        --------
        
            import ppm
            run1='/rpod3/fherwig/PPM/RUNS_DATA/VLTP_MESA_M0.542/C1'
            run2='/rpod3/fherwig/PPM/RUNS_DATA/sakurai-num-exp-robustness-onset-GOSH/A1/'
            YY=ppm.yprofile(run1)
            YY2=ppm.yprofile(run2)
            YY.tvmax(ifig=1,label='VLTP_0.543',id=0)
            YY2.tvmax(ifig=1,label='Sak_A1',id=1)
            
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
        #pl.rcParams.update(params)
        
        if ifig is None:
            pl.figure()
        else:
            pl.figure(ifig)
        
        t=self.get('t')
        EkMax=self.get('EkMax')

        vMax = 1000*np.sqrt(2.*EkMax)   # velocity in km.s
        if logy:
            y = np.log10(vMax)
            ylab = '$\log v_{\\rm max}/ {\\rm km\,s}^{-1}$'
        else:
            y = vMax
            ylab = '$v_{\\rm max} / {\\rm km\,s}^{-1}$'
        if label is not None:
            pl.plot(t/60,y,utils.linestyle(id)[0], markevery=utils.linestyle(id)[1],
                    label=label)
        else:
            pl.plot(t/60,y,utils.linestyle(id)[0], markevery=utils.linestyle(id)[1],
                    label=str(id))
        pl.legend(loc='best').draw_frame(False)
        pl.xlabel('t/min')
        pl.ylabel(ylab)
        if save:
            pl.savefig(prefix+'-t-vMax.'+format,format=format)

    def Richardson_plot(self, fname1 = 0, fname2 = 2, R_low = None, R_top = None, \
                        do_plots = False, logRi_levels = [-1., -0.6, 0., 1., 2., 3.], \
                        ylim_max = 2.02, compressible_fluid = True, plot_type = 0, \
                        ifig = 101):
        
        '''
        Make a plot of radius vs tangential velocity in the vicinity of the
        boundary and draw on lines of constant Richardson number. Compared to
        the function that produced Fig. 9 of Woodward+ (2015) this one takes
        into account the compressibility of the gas. Several bugs have been
        removed, too.
        

        Parameters
        ----------
        fname1 : int
            Which dump do you want to take the stratification from?
        fname2 : int
            Which dump do you want to take the velocities from?
        R_low : float
            The minimum radius in the plot. If invalid or None it will be set
            to R_top - 1.
        R_top : float
            Radius of top of convection zone. If invalid or None it will be set 
            to the radius at which FV H+He = 0.9.
        do_plots : logical
            Do you want to do some intermittent plotting?
        logRi_levels : list, optional
            Values of Ri for which to draw contours.
        ylim_max : float
            Max of ylim (min is automatically determined) in the final plot.
        compressible_fluid : logical
            You can set it to False to use the Richardson criterion for
            an incompressible fluid.
		plot_type : int
			plot_type = 0: Use a variable lower endpoint and a fixed upper endpoint of
                           the radial interval, in which Ri is calculated. Ri is plotted
						   for a range of assumed velocity differences with respect to
                           the upper endpoint.
			plot_type = 1: Compute Ri locally assuming that the local velocities vanish
                           on a certain length scale, which is computed from the radial
                           profile of the RMS horizontal velocity.
        ifig : int
            Figure number for the Richardson plot (a new window must be opened).
        '''
        
        # the whole calculation is done in code units
        # the unit of G in the code is 10^{-3} g cm^3 s^{-2}
        G_code = ast.grav_const/1e-3
        
        if fname1 < 0 or fname1 > np.max(list(self.ndumpDict.keys())):
            raise IOError("fname1 out of range.")
        
        if fname2 == 0:
            raise IOError("Velocities at fname2=0 will be 0; please "+\
                          "make another choice")
        
        if fname2 < 0 or fname2 > np.max(list(self.ndumpDict.keys())):
            raise IOError("fname2 out of range.")
        
        if plot_type != 0 and plot_type != 1:
            print("plot_type = %s is not implemented." % str(plot_type))
            return
        
        # get some header attributes
        R_bot = float(self.hattrs['At base of the convection zone R'])
        g_bot = float(self.hattrs['At base of the convection zone g'])
        
        # get the stratification at fname = fname1
        # radius is called 'Y' in the YProfiles
        r = self.get('Y', fname = fname1, resolution = 'l')
        fv_H_He = self.get('FV H+He', fname = fname1, resolution = 'l')
        rho = self.get('Rho', fname = fname1, resolution = 'l')
        p = self.get('P', fname = fname1, resolution = 'l')
        
        # get the rms horizontal velocities at fname = fname2
        ek_xz = self.get('EkXZ', fname = fname2, resolution = 'l')
        rms_u_xz = np.sqrt(ek_xz) # no factor of 2 for ek_xz (known bug)
        
        # pre-compute logarithms of some variables to speed up the code
        logrho = np.log(rho)
        logp = np.log(p)
        
        min_r = np.min(r)
        max_r = np.max(r)
        dr = cdiff(r)
        
        if R_top is not None:
            if R_top < min_r:
                print("R_top too low.")
                return
            elif R_top > max_r:
                print("R_top too high.")
                return
            else:
                # centre R_top on the nearest cell
                idx_top = np.argmin(np.abs(r - R_top))
                R_top = r[idx_top]
                print("R_top centred on the nearest cell: R_top = %.3f." % R_top)
        else:
            # put R_top where fv_H_He is the closest to 0.9
            idx_top = np.argmin(np.abs(fv_H_He - 0.9))
            R_top = r[idx_top]
            print("R_top set to %.3f." % R_top)
        
        if R_low is not None:
            if R_low < min_r:
                print("R_low too low.")
                return
            elif R_low >= R_top:
                print("R_low too high.")
                return
            else:
                # centre R_low on the nearest cell
                idx_low = np.argmin(np.abs(r - R_low))
                R_low = r[idx_low]
                print("R_low centred on the nearest cell: R_low = %.3f." % R_low)
        else:
            # the default setting
            R_low = R_top - 1.
            if R_low < min_r:
                R_low = min_r
            
            # find the point nearest to r = R_low
            idx_low = np.argmin(np.abs(r - R_low))
            R_low = r[idx_low]
            print("R_low centred on the cell nearest to R_top - 1: R_low = %.3f." % R_low)
        
        # centre R_bot on the nearest cell
        idx_bot = np.argmin(np.abs(r - R_bot))
        # mass below R_bot
        # (using r[idx_bot] instead of R_bot makes g[idx_bot] == g_bot)
        M_bot = g_bot*(r[idx_bot]**2)/G_code
        
        dm = 4.*np.pi*(r**2)*dr*rho
        m = np.cumsum(dm) # get the mass profile by integration
        # shift the mass profile to make sure that m[idx_bot] == M_bot
        # the mass profile at small radii won't make sense, because
        # the core is artificial with no gravity
        m += M_bot - m[idx_bot]
        
        g = G_code*m/(r**2) # gravity profile (see the note above)
        H_p = p/(rho*g) # pressure scale height (assuming hydrostatic equilibrium)
        
        nabla_ad = 0.4 # adiabatic temperature gradient
        nabla_rho_ad = 1. - nabla_ad # adiabatic density gradient
        
        # compute the Richardson number for the shearing flow
        # between r[idx_1] and r[idx_2] (it's faster when using
        # indices instead of radii)
        # arbitrary du or dudr can be supplied
        def Richardson(idx_1, idx_2, du = None, dudr = None):
            # average g and H_p between idx_1 and idx_2
            g_avg = (g[idx_1] + g[idx_2])/2.
            H_p_avg = (H_p[idx_1] + H_p[idx_2])/2.
            
            # approximate density gradient between idx_1 and idx_2
            dlogrho = logrho[idx_2] - logrho[idx_1]
            dlogp = logp[idx_2] - logp[idx_1]
            nabla_rho = dlogrho/dlogp
            
            # buoyancy frequency squared
            if compressible_fluid:
                N2 = (g_avg/H_p_avg)*(nabla_rho - nabla_rho_ad)
            else:
                N2 = (g_avg/H_p_avg)*nabla_rho
            
            # compute the velocity difference we have no information
            # about the velocity difference or gradient
            if du is None and dudr is None:
                du = rms_u_xz[idx_2] - rms_u_xz[idx_1]
            
            # compute the velocity gradient if none was supplied
            if dudr is None:
                dr = r[idx_2] - r[idx_1]
                dudr = du/dr
            
            # velocity gradient squared
            dudr2 = (dudr)**2
            
            # Richardson number
            Ri = N2/dudr2
            
            return Ri
        
        if plot_type == 0:
            # grid of indices for a sequence of intervals, in which Ri will be computed
            idx_grid = np.arange(idx_top + 1, idx_low + 1)
            
            # the corresponding grid of radii
            r_grid = np.array([r[i] for i in idx_grid])
        
            # reference velocity
            u_0 = rms_u_xz[idx_top]
            
            # velocity limit for the plot
            # the factor of 1e-3 converts the velocity from km/s to
            # code units (10^8 cm s^{-1})
            u_max = 1e-3*(10**ylim_max)
        
            # construct grids of assumed velocities and velocity differences
            # logarithmic spacing between (u_0 + log10u_step) and u_max
            log10u_step = 0.02
            u_grid = 10**np.arange(np.log10(rms_u_xz[idx_top]) + log10u_step, \
                     np.log10(u_max), log10u_step)
            du_grid = u_grid - u_0
            
            # perhaps the loops could be optimised, but it runs fast enough as is
            Ri = np.zeros((du_grid.size, idx_grid.size))
            for i in range(0, du_grid.size):
                for j in range(0, idx_grid.size):
                    Ri[i, j] = Richardson(idx_grid[j], idx_top, du = du_grid[i])
            
            pl.close(ifig)
            pl.figure(ifig)
            
            # pl.contour() fails if np.log10(Ri) is undefined everywhere
            if any(Ri > 0):
                cs = pl.contour(r_grid, np.log10(1e3*u_grid), np.log10(Ri), \
                                logRi_levels, linestyles = '-')
                pl.clabel(cs)
            
            # pl.contour() fails if np.log10(Ri) is undefined everywhere
            if any(Ri < 0):
                cs2 = pl.contour(r_grid, np.log10(1e3*u_grid), np.log10(-Ri), \
                                 logRi_levels, linestyles = '--')
                pl.clabel(cs2)
            
            pl.plot(r, np.log10(1e3*rms_u_xz + 1e-100), \
                    marker = 'o', markevery = utils.linestyle(0)[1], \
                    label = ("velocity at dump %d" % fname2))

            pl.xlabel(r'$r\ \mathrm{[Mm]}$', fontsize = 16)
            pl.ylabel(r"$\log\ u_\mathrm{hor,rms}\ \mathrm{[km/s]}$", fontsize = 16)

            pl.xlim(R_low, R_top)
            pl.ylim(np.log10(1e3*rms_u_xz[idx_top]), np.log10(1e3*u_max))
            pl.legend(loc = 3)
        elif plot_type == 1:
            # the steepest velocity gradient between R_low and R_top
            dudr = cdiff(rms_u_xz)/dr
            max_dudr = np.max(np.abs(dudr[idx_top:(idx_low + 1)]))
            
            # characteristic length scale on which velocities decrease at the boundary
            l_u = np.max(rms_u_xz[idx_top:(idx_low + 1)])/max_dudr
            print("Velocities decrease on a characteristic length scale of %.2f Mm" % l_u)
            
            # grid of positions, at which Ri will be computed
            idx_grid = np.arange(idx_top, idx_low + 1)
            
            # the corresponding grid of radii
            r_grid = np.array([r[i] for i in idx_grid])
            
            # estimate the local velocity gradient assuming that the local velocities
            # vanish on the length scale l_u
            dudr_estimate = rms_u_xz[idx_top:(idx_low + 1)]/l_u
            
            # compute the local Richardson numbers
            Ri = np.zeros(idx_grid.size)
            for i in range(0, idx_grid.size):
                Ri[i] = Richardson(idx_grid[i] - 1, idx_grid[i] + 1, dudr = dudr_estimate[i])

            # determine the upper limit for Ri in the plot
            min_log10absRi = np.min(np.log10(np.abs(Ri)))
            # always keep some margin
            if min_log10absRi - np.floor(min_log10absRi) > 0.1:
                pl_Ri_min = np.floor(min_log10absRi)
            else:
                pl_Ri_min = np.floor(min_log10absRi) - 1.
            
            # determine the upper limit for Ri in the plot
            max_log10absRi = np.max(np.log10(np.abs(Ri)))
            # always keep some margin
            if np.ceil(max_log10absRi) - max_log10absRi > 0.1:
                pl_Ri_max = np.ceil(max_log10absRi)
            else:
                pl_Ri_max = np.ceil(max_log10absRi) + 1.
            
            # FV values smaller than 10^{-8} are not interesting
            min_log10fv_H_He = np.min(np.log10(fv_H_He[idx_top:(idx_low + 1)] + 1e-100))
            if min_log10fv_H_He < -8.:
                min_log10fv_H_He = -8.
            
            # do we need to shift the FV curve in the plot?
            fv_offset = 0
            if pl_Ri_min > min_log10fv_H_He:
                fv_offset = pl_Ri_max
            
            pl.close(ifig)
            fig = pl.figure(ifig)
            ax = fig.add_subplot(111)
            lns = [] # array of lines to be put into a joint legend
            
            if any(Ri < 0):
                # temporarily suppress numpy warnings
                old_settings = np.seterr()
                np.seterr(invalid = 'ignore')
                lns += ax.plot(r_grid, np.log10(Ri), linestyle = '-', linewidth = 2, \
                               label = r'$Ri > 0$')
                lns += ax.plot(r_grid, np.log10(-Ri), linestyle = '--', linewidth = 2, \
                               label = r'$Ri < 0$')
                np.seterr(**old_settings)
            else:
                lns += ax.plot(r_grid, np.log10(Ri), linestyle = '-', linewidth = 2, \
                               label = r'$Ri$')
            
            lns += ax.plot(r_grid, np.log10(fv_H_He[idx_top:(idx_low + 1)] + 1e-100) + \
                           fv_offset, linestyle = '-', label = r'$FV$')
            
            ax.set_xlim(R_low, R_top)
            ax.set_ylim(pl_Ri_min, pl_Ri_max)
            
            ax.set_xlabel(r'$\mathrm{radius\ [Mm]}$', fontsize = 16)
            if fv_offset == 0:
                ax.set_ylabel(r'$\log Ri; \log FV$', fontsize = 16)
            else:
                ax.set_ylabel(r'$\log Ri;\ \log FV + %d$' % fv_offset, fontsize = 16)

            ax2 = ax.twinx() # get a new set of axes
            lns += ax2.plot(r_grid, 1e3*rms_u_xz[idx_top:(idx_low + 1)], linestyle = '-.', \
                            label = r'$u_\mathrm{hor,rms}$')
            ax2.set_ylabel(r'$u_\mathrm{hor,rms}\ \mathrm{[km/s]}$', fontsize = 16)

            lbls = [l.get_label() for l in lns]
            ax2.legend(lns, lbls, loc = 2)
        
        # show some plots to inspect more variables
        if do_plots:
            i1 = idx_bot
            i2 = np.argmax(r)
            if i1 > i2: tmp = i1; i1 = i2; i2 = tmp
            
            pl.close(102); pl.figure(102)
            pl.plot(r[i1:i2], 5.025e-07*m[i1:i2])
            pl.xlabel("radius [Mm]")
            pl.ylabel("mass [M_Sun]")
            pl.title("enclosed mass")
            
            pl.close(103); pl.figure(103)
            pl.plot(r[i1:i2], 1e8*g[i1:i2])
            pl.xlabel("radius [Mm]")
            pl.ylabel("gravity [cm/s^2]")
            pl.title("gravity")
            
            pl.close(104); pl.figure(104)
            pl.plot(r[i1:i2], np.log10(1e3*rho[i1:i2]))
            pl.xlabel("radius [Mm]")
            pl.ylabel("log10(rho [g/cm^3])")
            pl.title("log10 density")
            
            pl.close(105); pl.figure(105)
            pl.plot(r[i1:i2], np.log10(1e19*p[i1:i2]))
            pl.xlabel("radius [Mm]")
            pl.ylabel("log10(p [g/(cm s^2)])")
            pl.title("log10 pressure")
            
            pl.close(106); pl.figure(106)
            pl.plot(r[i1:i2], H_p[i1:i2])
            pl.xlabel("radius [Mm]")
            pl.ylabel("Hp [Mm]")
            pl.title("pressure scale height")
            
            pl.close(107); pl.figure(107)
            nabla_rho = cdiff(logrho)/cdiff(logp)
            pl.plot(r[i1:i2], nabla_rho[i1:i2])
            pl.xlabel("radius [Mm]")
            pl.ylabel("nabla_rho")
            pl.title("density gradient")
            
            pl.close(108); pl.figure(108)
            nabla_rho = cdiff(logrho)/cdiff(logp)
            N2 = (g/H_p)*(nabla_rho - nabla_rho_ad)
            pl.plot(r[i1:i2], N2[i1:i2])
            pl.xlabel("radius [Mm]")
            pl.ylabel("N^2 [1/s^2]")
            pl.title("buoyancy frequency")
        
        pl.show() # show everything we have plotted

    def Dov(self,r0,D0,f,fname=1):
        '''
        Calculate and plot an exponentially decaying diffusion coefficient
        given an r0, D0 and f.
            
        Dov is given by the formula D0*exp(-2*(r-r0)/f*Hp)
            
        Parameters
        ----------
        r0 : float
            radius (Mm) at which the decay will begin
        D0 : float
            diffusion coefficient at r0
        f : float
            what f-value (parameter, where f*Hp is the e-folding length
            of the diffusion coefficient) should we decay with?
        fname : int
            which dump do you want to take r and P from?
            
        Output
        ------
        r : array
            radial co-ordinates (Mm) for which we have a diffusion coefficient
            the decays exponentially
        D : array
            Exponentially decaying diffusion coefficient (cm^2/s)
        '''
        #    Hp = 1 / d lnP/dr = P / (d)
        r = self.get('Y',fname=fname,resolution='l')[::-1] * 1.e8  # cm, centre to surface
        P = self.get('P',fname=fname)[::-1] * 1.e19 # barye, centre to surface
        Hp = - P[1:] * np.diff(r) / np.diff(P)
        Hp = np.insert(Hp,0,0)
        idx = np.abs(r - r0*1.e8).argmin()
        r0 = r[idx]
        Hp0 = Hp[idx]
        
        print(r0, Hp0, idx)
        
        D = D0 * np.exp(-2. * (r[idx:] - r0) / f / Hp0)
        return r[idx:] / 1.e8, D

    def Dov2(self,r0,D0,f1=0.2,f2=0.05,fname=1, silent = True):
        '''
        Calculate and plot an 2-parameter exponentially decaying diffusion coefficient
        given an r0, D0 and f1 and f2.
        
        Dov is given by the formula:
            D = 2. * D0 * 1. / (1. / exp(-2*(r-r0)/f1*Hp) + 
                               1. / exp(-2*(r-r0)/f2*Hp))
        
        Parameters
        ----------
        r0 : float
            radius (Mm) at which the decay will begin
        D0 : float
            diffusion coefficient at r0
        f1,f2,A : float
            parameters of the model
        fname : int
            which dump do you want to take r and P from?
        
        Output
        ------
        r : array
            radial co-ordinates (Mm) for which we have a diffusion coefficient
            the decays exponentially
        D : array
            Exponentially decaying diffusion coefficient (cm^2/s)
        '''
        #    Hp = 1 / d lnP/dr = P / (d)
        r = self.get('Y',fname=fname,resolution='l')[::-1] * 1.e8  # cm, centre to surface
        P = self.get('P',fname=fname)[::-1] * 1.e19 # barye, centre to surface
        Hp = - P[1:] * np.diff(r) / np.diff(P)
        Hp = np.insert(Hp,0,0)
        idx = np.abs(r - r0*1.e8).argmin()
        r0 = r[idx]
        Hp0 = Hp[idx]
        if not silent:
            print(r0, Hp0, idx)
                
        D = 2. * D0 * 1./(1./(np.exp(-2. * (r - r0) / f1 / Hp0)) +
                         1./(np.exp(-2. * (r - r0) / f2 / Hp0))
                         )
        return r / 1.e8, D
    
    def Dinv(self,fname1,fname2,fluid='FV H+He',numtype='ndump',newton=False,
             niter=3,debug=False,grid=False,FVaverage=False,tauconv=None,
             returnY=False,plot_Dlt0=True, silent = True, initial_conv_boundaries = True,
             approx_D=False,linelims=None, r0 = None):
        '''
        Solve inverted diffusion equation to see what diffusion coefficient
        profile would have been appropriate to mimic the mixing of species
        seen in the Yprofile dumps.
            
        In the present version, we only solve for D in the region where
        'FV H+He' is not 0 or 1 in dump fname 2.
            
        Parameters
        ----------
        fname1,fname2 : int or float
            cycles from which to take initial and final abundance profiles
            for the diffusion step we want to mimic.
        fluid : string
            Which fluid do you want to track?
        numtype : string, optional
            Designates how this function acts and how it interprets
            fname.  If numType is 'file', this function will get the
            desired attribute from that file.  If numType is 'NDump'
            function will look at the cycle with that nDump.  If
            numType is 'T' or 'time' function will find the _cycle
            with the closest time stamp.
            The default is "ndump".
        newton : boolean, optional
            Whether or not to apply Newton-Raphson refinement of the
            solution for D.
            The default is False
        niter : int, optional
            If N-R refinement is to be done, niter is how many iterations
            to compute.
            The default is 3.
        grid : boolean, optional
            whether or not to show the axes grids.
            The default is False.
        FVaverage : boolean, optional
            Whether or not to average the abundance profiles over a
            convective turnover timescale. See also tauconv.
            The default is False.
        tauconv : float, optional
            If averaging the abundance profiles over a convective turnover
            timescale, give the convective turnover timescale (seconds).
            The default value is None.
        returnY : boolean, optional
            If True, return abundance vectors as well as radius and diffusion
            coefficient vectors
            The default is False.
        plot_Dlt0 : boolean, optional
            whether or not to plot the diffusion coefficient when it is
            negative
        approx_D : boolean, optional
            whether or not to plot an approximate diffusion coefficient 
            for an area of the line
        linelims : range, optional
            limits of the radius to approximate diffusion coefficient
            default is none (whole vector)
        r0 = None, optional
            Start of exponential diffusion decay, necessary 
            for approx_D
            
        Output
        ------
        x : array
            radial co-ordinates (Mm) for which we have a diffusion coefficient
        D : array
            Diffusion coefficient (cm^2/s)
            
        Example
        -------
        YY=ppm.yprofile(path_to_data)
        YY.Dinv(1,640)
        '''


        xlong = self.get('Y',fname=fname1,resolution='l') # for plotting
        if debug: print(xlong)
        x = xlong
        x = x * 1.e8

        def mf(fluid,fname):
            '''
            Get mass fraction profile of fluid 'fluid' at fname.
            '''
            y = self.get(fluid,fname=fname,resolution='l')
            if fluid == 'FV H+He':
                rhofluid = self.get('Rho H+He',fname=fname,resolution='l')
            else:
                rhofluid = self.get('RHOconv',fname=fname,resolution='l')
            rho = self.get('Rho',fname=fname,resolution='l')
            y = rhofluid * y / rho
            return y
    
        if FVaverage is False:
            y1 = mf(fluid,fname2)
            y1long = y1 # for plotting

            y0 = mf(fluid,fname1)
            y0long = y0 # for plotting
        else:
            if tauconv is None:
                raise IOError("Please define tauconv")
            # Find the dumps accross which one should average:
            # first profile:
            myt0 = self.get('t',fname1)[-1]
            myt01 = myt0 - tauconv / 2.
            myt02 = myt0 + tauconv / 2.
            myidx01 = np.abs(self.get('t') - myt01).argmin()
            myidx02 = np.abs(self.get('t') - myt02).argmin()
            mycyc01 = self.cycles[myidx01]
            mycyc02 = self.cycles[myidx02]
            # second profile:
            myt1 = self.get('t',fname2)[-1]
            myt11 = myt1 - tauconv / 2.
            myt12 = myt1 + tauconv / 2.
            myidx11 = np.abs(self.get('t') - myt11).argmin()
            myidx12 = np.abs(self.get('t') - myt12).argmin()
            mycyc11 = self.cycles[myidx11]
            mycyc12 = self.cycles[myidx12]
            # do the average for the first profile:
            ytmp = np.zeros(len(x))
            count=0
            for cyc in range(mycyc01,mycyc02):
                ytmp += mf(fluid,cyc)
                count+=1
            y0 = ytmp / float(count)
            # do the average for the first profile:
            ytmp = np.zeros(len(x))
            count=0
            for cyc in range(mycyc11,mycyc12):
                ytmp += mf(fluid,cyc)
                count+=1
            y1 = ytmp / float(count)
    
            y0long = y0
            y1long = y1

        if fluid == 'FV H+He':
            y1 = y1[::-1]
            x = x[::-1]
            y0 = y0[::-1]

        if debug: print(len(xlong), len(y0long))

        idx0 = np.abs(np.array(self.cycles) - fname1).argmin()
        idx1 = np.abs(np.array(self.cycles) - fname2).argmin()
        t0 = self.get('t')[idx0]
        t1 = self.get('t')[idx1]
        deltat = t1 - t0
        
        # now we want to exclude any zones where the abundances
        # of neighboring cells are the same. This is hopefully
        # rare inside the computational domain and limited to only
        # a very small number of zones
        indexarray = np.where(np.diff(y1) == 0)[0]
        if not silent:
            print('removing zones:', indexarray)
        y1 = np.delete(y1,indexarray)
        y0 = np.delete(y0,indexarray)
        x = np.delete(x,indexarray)
        
        # in the current formulation for the inner boundary condition,
        # y1[0] != 0:
        while y1[0] == 0.:
            x = x[1:]
            y0 = y0[1:]
            y1 = y1[1:]
                
        # Try moving left boundary one over to allow for "exact"
        # boundary condition and saving the now ghost cell value
        xl = x[100]
        y0l = y0[100]
        y1l = y1[100]
        x = x[101:]
        y0 = y0[101:]
        y1 = y1[101:]

        if debug : print(y0, y1, deltat)
        if not silent:
            print('deltat = ', deltat, 's')
        p = np.zeros(len(x))
        q = np.zeros(len(x))
        
        xdum = np.zeros(3) # our workhorse array for differencing
        
        dt = float(deltat)
        
        # calculate matrix elements for intermediate mesh points:
        def matrixdiffus(x, y0, y1, dt):
            m = len(x) - 1
            for i in range(1,m):
                xdum[0] = x[i-1]
                xdum[1] = x[i]
                xdum[2] = x[i+1]
                
                xl = xdum[1] - xdum[0]
                xr = xdum[2] - xdum[1]
                
                xm = 0.5 * (xdum[2] - xdum[0])
                
                alpha = dt / xm
                
                p[i] = (y1[i] - y1[i-1]) * alpha / xl
                q[i] = (y1[i] - y1[i+1]) * alpha / xr
            
            # central (left) boundary:
            xdum[1] = x[0]
            xdum[2] = x[1]
            xr = xdum[2] - xdum[1]
            alpha = dt / (xr * xr)
#            p[0] = y1[i] * alpha
#            q[0] = (y1[i] - y1[i+1]) * alpha
#            p[0] = y1[0] * alpha
#            q[0] = (y1[0] - y1[1]) * alpha
            # Trying new boundary conditions:
            p[0] = (y1[0] - y1l) * alpha
            q[0] = (y1[0] - y1[1]) * alpha
            if not silent:
                print('p0, q0 = ', p[0],q[0])
            
            # surface (right) boundary:
            xdum[0] = x[m-1]
            xdum[1] = x[m]
            xl = xdum[1] - xdum[0]
            alpha = dt / (xl * xl)
#            p[m] = (y1[i] - y1[i-1]) * alpha
#            q[m] = 0
            p[m] = (y1[m] - y1[m-1]) * alpha
            q[m] = 0.
            if not silent:
                print('pm, qm = ', p[m],q[m])
            
            G = np.zeros([len(x),len(x)])
            
            # set up matrix:
            
            for i in range(len(x)):
                if not silent:
                    print('p[i] = ', p[i])
                G[i,i] = p[i]
                if debug : print(G[i,i])
                if i != len(x)-1 :
                    G[i,i+1] = q[i]
                    if debug : print(G[i,i+1])
        
            A = np.array( [ G[i,:] for i in range(len(x)) ] )
            if not silent:
                print(A[0])
                print('determinant = ', np.linalg.det(A))
            return A
        
        
        # Direct solution (initial guess if moving on to do Newton-
        # Raphson refinement:
        A = matrixdiffus(x,y0,y1,dt)
        B = y0 - y1
        D = np.linalg.solve(A,B)
        
        if newton:
            x0 = D
            xn = np.zeros(len(x0))
            xn = x0
            xnp1 = np.zeros(len(x0))
            J = np.linalg.inv(A) # Jacobian matrix
            # refinement loop:
            for i in range(1,niter+1):
                f = np.dot(A,xn) - B
                xnp1 = xn - np.dot(J,f)
                corr = np.abs(xnp1 - xn) / xn
                cmax = max(corr)
                cmin = min(corr)
                if not silent:
                    print('NEWTON: iter '+str(i))
                    print('max. correction = ', cmax)
                    print('min. correction = ', cmin)
                xn = xnp1
            
            D = xnp1

        cb = utils.colourblind
        lsty = utils.linestyle
        
        pl.figure()
        pl.plot(xlong,np.log10(y0long),\
                marker='o',
                color=cb(8),\
                markevery=lsty(1)[1],\
                label='$X_{'+str(fname1)+'}$')
        pl.plot(xlong,np.log10(y1long),\
                marker='o',\
                color=cb(9),\
                markevery=lsty(2)[1],\
                label='$X_{'+str(fname2)+'}$')
#        pl.ylabel('$\log\,X$ '+fluid.replace('FV',''))
        pl.ylabel('$\log\,X$ ')
        pl.xlabel('r / Mm')
        pl.ylim(-8,0.1)
        pl.legend(loc='center right').draw_frame(False)
        if grid:
            pl.grid()
        pl.twinx()
        pl.plot(x/1.e8,np.log10(D),'k-',\
                label='$D$') #'$D > 0$')
        if initial_conv_boundaries:
                pl.axvline(self.radbase,linestyle='dashed',color='k')
                pl.axvline(self.radtop,linestyle='dashed',color='k')
        if plot_Dlt0:
            pl.plot(x/1.e8,np.log10(-D),'k--',\
                    label='$D < 0$')
        pl.ylabel('$\log(D\,/\,{\\rm cm}^2\,{\\rm s}^{-1})$')
        if approx_D:
            if linelims is not None:
                indx1 = np.abs(x/1.e8 - linelims[0]).argmin()
                indx2 = np.abs(x/1.e8 - linelims[1]).argmin()
                m,b = pyl.polyfit(x[indx1:indx2]/1.e8,np.log10(D[indx1:indx2]),1)
            else:
                m,b = pyl.polyfit(x/1.e8,np.log10(D),1)
            if r0 is None:
                print('Please define r0')
            rr = self.get('Y',fname=1,resolution='l')[::-1]
            P = self.get('P',fname=fname1)[::-1] * 1.e19 # barye, centre to surface
            Hp = - P[1:] * np.diff(rr) / np.diff(P)
            Hp = np.insert(Hp,0,0)
            idxhp = np.abs(rr - r0).argmin()
            idx = np.abs(x - r0*1.e8).argmin()
            Hp0 = Hp[idxhp]
            D0 = D[idx]
            f = (-2. * np.abs(np.mean(x[indx1:indx2])/1.e8 - r0))\
                /( np.log( (np.mean(D[indx1:indx2]))/D0) *Hp0)
            lab='f= '+str("%.3f" % f)
            pl.plot(x[indx1:indx2]/1.e8,m*x[indx1:indx2]\
                    /1.e8+b,linestyle='dashed',color='r',label=lab)
        pl.legend(loc='upper right').draw_frame(False)
        if returnY:
            return x/1.e8, D, y0, y1
        else:
            return x/1.e8,D

    def Dsolve(self,fname1,fname2,fluid='FV H+He',numtype='ndump',newton=False,niter=3,
             debug=False,grid=False,FVaverage=False,tauconv=None,returnY=False):
        '''
            Solve inverse diffusion equation sequentially by iterating over the spatial
            domain using a lower boundary condition (see MEB's thesis, page
            223, Eq. B.15).
            
            
            Parameters
            ----------
            fname1,fname2 : int or float
                cycles from which to take initial and final abundance profiles
                for the diffusion step we want to mimic.
            fluid : string
                Which fluid do you want to track?
                numtype : string, optional
                Designates how this function acts and how it interprets
                fname.  If numType is 'file', this function will get the
                desired attribute from that file.  If numType is 'NDump'
                function will look at the cycle with that nDump.  If
                numType is 'T' or 'time' function will find the _cycle
                with the closest time stamp.
                The default is "ndump".
            newton : boolean, optional
                Whether or not to apply Newton-Raphson refinement of the
                solution for D.
                The default is False
            niter : int, optional
                If N-R refinement is to be done, niter is how many iterations
                to compute.
                The default is 3.
            grid : boolean, optional
                whether or not to show the axes grids.
                The default is False.
            FVaverage : boolean, optional
                Whether or not to average the abundance profiles over a
                convective turnover timescale. See also tauconv.
                The default is False.
            tauconv : float, optional
                If averaging the abundance profiles over a convective turnover
                timescale, give the convective turnover timescale (seconds).
                The default value is None.
            returnY : boolean, optional
                If True, return abundance vectors as well as radius and diffusion
                coefficient vectors
                The default is False.
            
            Output
            ------
            x : array
                radial co-ordinates (Mm) for which we have a diffusion coefficient
            D : array
                Diffusion coefficient (cm^2/s)
            
            Example
            -------
            YY=ppm.yprofile(path_to_data)
            YY.Dsolve(1,640)
        '''
        
        
        xlong = self.get('Y',fname=fname1,resolution='l') # for plotting
        if debug: print(xlong)
        x = xlong
        x = x * 1.e8
        
        def mf(fluid,fname):
            '''
                Get mass fraction profile of fluid 'fluid' at fname.
                '''
            y = self.get(fluid,fname=fname,resolution='l')
            if fluid == 'FV H+He':
                rhofluid = self.get('Rho H+He',fname=fname,resolution='l')
            else:
                rhofluid = self.get('RHOconv',fname=fname,resolution='l')
            rho = self.get('Rho',fname=fname,resolution='l')
            y = rhofluid * y / rho
            return y
        
        if FVaverage is False:
            y1 = mf(fluid,fname2)
            y1long = y1 # for plotting
            
            y0 = mf(fluid,fname1)
            y0long = y0 # for plotting
        else:
            if tauconv is None:
                raise IOError("Please define tauconv")
            # Find the dumps accross which one should average:
            # first profile:
            myt0 = self.get('t',fname1)[-1]
            myt01 = myt0 - tauconv / 2.
            myt02 = myt0 + tauconv / 2.
            myidx01 = np.abs(self.get('t') - myt01).argmin()
            myidx02 = np.abs(self.get('t') - myt02).argmin()
            mycyc01 = self.cycles[myidx01]
            mycyc02 = self.cycles[myidx02]
            # second profile:
            myt1 = self.get('t',fname2)[-1]
            myt11 = myt1 - tauconv / 2.
            myt12 = myt1 + tauconv / 2.
            myidx11 = np.abs(self.get('t') - myt11).argmin()
            myidx12 = np.abs(self.get('t') - myt12).argmin()
            mycyc11 = self.cycles[myidx11]
            mycyc12 = self.cycles[myidx12]
            # do the average for the first profile:
            ytmp = np.zeros(len(x))
            count=0
            for cyc in range(mycyc01,mycyc02):
                ytmp += mf(fluid,cyc)
                count+=1
            y0 = ytmp / float(count)
            # do the average for the first profile:
            ytmp = np.zeros(len(x))
            count=0
            for cyc in range(mycyc11,mycyc12):
                ytmp += mf(fluid,cyc)
                count+=1
            y1 = ytmp / float(count)
            
            y0long = y0
            y1long = y1
        
        if fluid == 'FV H+He':
            y1 = y1[::-1]
            x = x[::-1]
            y0 = y0[::-1]
        
        if debug: print(len(xlong), len(y0long))
        
        idx0 = np.abs(np.array(self.cycles) - fname1).argmin()
        idx1 = np.abs(np.array(self.cycles) - fname2).argmin()
        t0 = self.get('t')[idx0]
        t1 = self.get('t')[idx1]
        deltat = t1 - t0
        
        # now we want to exclude any zones where the abundances
        # of neighboring cells are the same. This is hopefully
        # rare inside the computational domain and limited to only
        # a very small number of zones
        indexarray = np.where(np.diff(y1) == 0)[0]
        print('removing zones:', indexarray)
        y1 = np.delete(y1,indexarray)
        y0 = np.delete(y0,indexarray)
        x = np.delete(x,indexarray)
    
        # in the current formulation for the inner boundary condition,
        # y1[0] != 0:
        while y1[0] == 0.:
            x = x[1:]
            y0 = y0[1:]
            y1 = y1[1:]
        
        # Try moving left boundary one over to allow for "exact"
        # boundary condition and saving the now ghost cell value
        xl = x[0]
        y0l = y0[0]
        y1l = y1[0]
        x = x[1:]
        y0 = y0[1:]
        y1 = y1[1:]
        
        if debug : print(y0, y1, deltat)
        print('deltat = ', deltat, 's')
        p = np.zeros(len(x))
        q = np.zeros(len(x))
        
        xdum = np.zeros(3) # our workhorse array for differencing
        
        dt = float(deltat)
        
        # Calculate D starting from inner boundary:
        D = np.zeros(len(x))
        # inner boundary:
        xr = x[1] - x[0]
        xl = xr
        xm = (xl + xr) / 2.
        p = (y1[0] - y1l) / xl
        s = xm * (y1[0] - y0[0]) / dt
        u = xr / (y1[1] - y1[0])
        Dghost = 0. # is this OK?
        D[0] = u * (s + Dghost * p)
        # now do the rest:
        for i in range(1,len(x)-1):
            xr = x[i+1] - x[i]
            xl = x[i] - x[i-1]
            xm = (xl + xr) / 2.
            p = (y1[i] - y1[i-1]) / xl
            s = xm * (y1[i] - y0[i]) / dt
            u = xr / (y1[i+1] - y1[i])
            D[i] = u * (s + D[i-1] * p)
        # outer boundary:
        m = len(x) - 1
        xl = x[m] - x[m-1]
        xr = xl
        xm = (xl + xr) / 2.
        p = (y1[m] - y1[m-1]) / xl
        s = xm * (y1[m] - y0[m]) / dt
        u = xr / (1. - y1[m]) # assuming here that y[m+1] - 1.. is this OK??
        D[m] = u * (s + D[m-1] * p)

        pl.figure()
        #        pl.plot(xlong,np.log10(y0long),utils.linestyle(1)[0],\
        #                markevery=utils.linestyle(1)[1],\
        #                label=fluid.replace('FV','')+' '+str(fname1))
        #        pl.plot(xlong,np.log10(y1long),utils.linestyle(2)[0],\
        #                markevery=utils.linestyle(2)[1],\
        #                label=fluid.replace('FV','')+' '+str(fname2))
        pl.plot(xlong,np.log10(y0long),utils.linestyle(1)[0],\
                markevery=utils.linestyle(1)[1],\
                label='fluid above'+' '+str(fname1))
        pl.plot(xlong,np.log10(y1long),utils.linestyle(2)[0],\
                markevery=utils.linestyle(2)[1],\
                label='fluid above'+' '+str(fname2))
        pl.ylabel('$\log\,X$ '+fluid.replace('FV',''))
        pl.xlabel('r / Mm')
        pl.ylim(-8,0.1)
        pl.legend(loc='lower left').draw_frame(False)
        if grid:
            pl.grid()
        pl.twinx()
        pl.plot(x/1.e8,np.log10(D),'k-',\
                label='$D$') #'$D > 0$')
        pl.plot(x/1.e8,np.log10(-D),'k--',\
                label='$D < 0$')
        pl.ylabel('$\log D\,/\,{\\rm cm}^2\,{\\rm s}^{-1}$')
        pl.legend(loc='upper right').draw_frame(False)
        
        if returnY:
            return x/1.e8, D, y0, y1
        else:
            return x/1.e8,D

    def Dsolvedown(self,fname1,fname2,fluid='FV H+He',numtype='ndump',
       newton=False,niter=3,debug=False,grid=False,FVaverage=False,
       tauconv=None,returnY=False,smooth=False,plot_Dlt0=True,
       sinusoidal_FV=False, log_X=True, Xlim=None, Dlim=None,
       silent=True,showfig=True):
        '''
        Solve diffusion equation sequentially by iterating over the spatial
        domain inwards from the upper boundary.        
        
        Parameters
        ----------
        fname1,fname2 : int or float
            cycles from which to take initial and final abundance profiles
            for the diffusion step we want to mimic.
        fluid : string
            Which fluid do you want to track?
        numtype : string, optional
            Designates how this function acts and how it interprets
            fname.  If numType is 'file', this function will get the
            desired attribute from that file.  If numType is 'NDump'
            function will look at the cycle with that nDump.  If
            numType is 'T' or 'time' function will find the _cycle
            with the closest time stamp.
            The default is "ndump".
        newton : boolean, optional
            Whether or not to apply Newton-Raphson refinement of the
            solution for D.
            The default is False
        niter : int, optional
            If N-R refinement is to be done, niter is how many iterations
            to compute.
            The default is 3.
        grid : boolean, optional
            whether or not to show the axes grids.
            The default is False.
        FVaverage : boolean, optional
            Whether or not to average the abundance profiles over a
            convective turnover timescale. See also tauconv.
            The default is False.
        tauconv : float, optional
            If averaging the abundance profiles over a convective turnover
            timescale, give the convective turnover timescale (seconds).
            The default value is None.
        returnY : boolean, optional
            If True, return abundance vectors as well as radius and diffusion
            coefficient vectors
            The default is False.
        smooth : boolean, optional
            Smooth the abundance profiles with a spline fit, enforcing their
            monotonicity. Only works for FV H+He choice of fluid
        plot_Dlt0 : boolean, optional
            whether or not to plot D where it is <0
            the default value is True
        
        Output
        ------
        x : array
            radial co-ordinates (Mm) for which we have a diffusion coefficient
        D : array
            Diffusion coefficient (cm^2/s)
        
        Example
        -------
        YY=ppm.yprofile(path_to_data)
        YY.Dsolvedown(1,640)
        '''
    
    
        xlong = self.get('Y',fname=fname1,resolution='l') # for plotting
        if debug: print(xlong)
        x = xlong
        
        def mf(fluid,fname):
            '''
                Get mass fraction profile of fluid 'fluid' at fname.
            '''
            y = self.get(fluid,fname=fname,resolution='l')
            if fluid == 'FV H+He':
                rhofluid = self.get('Rho H+He',fname=fname,resolution='l')
            else:
                rhofluid = self.get('RHOconv',fname=fname,resolution='l')
            rho = self.get('Rho',fname=fname,resolution='l')
            y = rhofluid * y / rho
            return y
        
        def make_monotonic(r,x):
            '''
            function for making x monotonic, as the solution to the diffusion
            equation should be. Only works when considering FV H+He (not conv)
            '''
            from scipy.interpolate import UnivariateSpline as uvs

            xorig = x
            rorig = r
            xnew0 = np.zeros(len(x))

            # find top and botom of convection zone

            ib = np.where(x!=0)[0][-1]
            it = np.where(x!=1)[0][0]

            x = x[it:ib]
            r = r[it:ib]
            x = np.maximum(x,1.e-12)
            x = np.log10(x)

            # is there a local min?

            dx = np.diff( x )
            c = len( np.where( dx > 0 )[0] ) > 0

            # find midpoint, left and right sides of linear reconstruction

            if c:
                # there is a local minimum, but also a local maximum
                # find the min by going top inwards and seeing where
                # difference changes sign
                # continue on to find the max, with which we set the width of the linear
                # reconstruction region
#                for i in range(len(r)):
#                    if dx[i] > 0.:
#                        idxmin = i
#                        break
# ask user for now to identify the local min (noisy data)
                pl.plot(r,x)
                usr_rmin = input("Please input local min in Mm: ")
                usr_rmin = float(usr_rmin)
                idxmin = np.abs( r - usr_rmin ).argmin()
                for i in range(idxmin,len(r)):
                    if x[i] == max(x[idxmin:]):
                        idxmax = i
                        break
            else:
                # everything was OK-ish anyway 
                return xorig

            # to take local max as left side as interval:
            if False:
                il = idxmax

            # set left side of interval to radius where x drops below the local minimum:
            if True:
                for i in range(idxmin,len(r)):
                    if x[i] < x[idxmin]:
                        il = i
                        break

            rmid = r[idxmin]
            width = ( rmid - r[il] ) * 2.
            rl = rmid - width / 2.
            rr = rmid + width / 2.

            il = np.abs( r - rl ).argmin()
            ir = np.abs( r - rr ).argmin()

            # just sort the two ends

            rt = r[:ir]
            rb = r[il:]

            xt = np.array(sorted(x[:ir])[::-1])
            xb = np.array(sorted(x[il:])[::-1])

            # now we fit the reconstruction region

            def expfunc(x, a, c, d):
                return a*np.exp(c*x)+d

            if True:

                rm = r[ir:il]
                xms = sorted(x[ir:il])[::-1]
                xms = np.array(xms)
                from scipy.optimize import curve_fit
                # fit an exponential
                #popt, pcov = curve_fit(expfunc, rm, xms, p0=(1, 1, 1))
                # linear reconstruction
                m = ( xms[-1] - xms[0] ) / ( rm[-1] - rm[0] )
                c = xms[0] - m * rm[0]
                # now extend it <bw> Mm beyond the l and r bounds of the reconstruction
                # region so that we can do a blend
                bw = 0.1
                idxr = np.abs( r - ( rr + bw ) ).argmin()
                idxl = np.abs( r - ( rl - bw ) ).argmin()
                rm = r[idxr:idxl]
                # exponential:
                #xm = func(rm, *popt)
                # linear:
                xm = m * rm + c
                            
            #now combine back the results with a sinusoidal blend at each overlap of the
            # reconstruction region with the top and bottom components

            xnew = np.zeros(len(x))
            # top
            itr = np.abs(rt-(rr+bw)).argmin()
            xnew[:idxr] = xt[:itr]
            # bottom
            ibl = np.abs(rb-(rl-bw)).argmin()
            xnew[idxl:] = xb[ibl:]
            # upper middle
            imrbb = np.abs( rm - rr ).argmin()
            xnew[idxr:ir] = xt[itr:] * np.sin( np.abs(rt[itr:] - r[ir]) / bw * np.pi / 2. ) ** 2 + \
                            xm[:imrbb] * np.cos( np.abs(rm[:imrbb] - r[ir]) / bw * np.pi / 2. ) ** 2
            # lower middle
            imlbt = np.abs( rm - rl ).argmin()
            xnew[il:idxl] = xb[:ibl] * np.sin( np.abs(rb[:ibl] - r[il]) / bw * np.pi / 2. ) ** 2 + \
                            xm[imlbt:] * np.cos( np.abs(rm[imlbt:] - r[il]) / bw * np.pi / 2. ) ** 2
            # middle
            xnew[ir:il] = xm[imrbb:imlbt]

            xnew0[it:ib] = xnew[:]
            xnew0 = 10. ** xnew0
            xnew0[:it] = 1.
            xnew0[ib:] = 0.
            return xnew0


        if FVaverage is False:
            y1 = mf(fluid,fname2)
            y1long = y1 # for plotting
            
            y0 = mf(fluid,fname1)
            y0long = y0 # for plotting
        else:
            if tauconv is None:
                raise IOError("Please define tauconv")
            # Find the dumps accross which one should average:
            # first profile:
            myt0 = self.get('t',fname1)[-1]
            myt01 = myt0 - tauconv / 2.
            myt02 = myt0 + tauconv / 2.
            myidx01 = np.abs(self.get('t') - myt01).argmin()
            myidx02 = np.abs(self.get('t') - myt02).argmin()
            mycyc01 = self.cycles[myidx01]
            mycyc02 = self.cycles[myidx02]
            # second profile:
            myt1 = self.get('t',fname2)[-1]
            myt11 = myt1 - tauconv / 2.
            myt12 = myt1 + tauconv / 2.
            myidx11 = np.abs(self.get('t') - myt11).argmin()
            myidx12 = np.abs(self.get('t') - myt12).argmin()
            mycyc11 = self.cycles[myidx11]
            mycyc12 = self.cycles[myidx12]
            # do the average for the first profile:
            ytmp = np.zeros(len(x))
            count=0
            for cyc in range(mycyc01,mycyc02):
                ytmp += mf(fluid,cyc)
                count+=1
            y0 = ytmp / float(count)
            # do the average for the first profile:
            ytmp = np.zeros(len(x))
            count=0
            for cyc in range(mycyc11,mycyc12):
                ytmp += mf(fluid,cyc)
                count+=1
            y1 = ytmp / float(count)
            
            y0long = y0
            y1long = y1
        
        
        if debug: print(len(xlong), len(y0long))
        
        idx0 = np.abs(np.array(self.cycles) - fname1).argmin()
        idx1 = np.abs(np.array(self.cycles) - fname2).argmin()
        t0 = self.get('t')[idx0]
        t1 = self.get('t')[idx1]
        deltat = t1 - t0
        
        if smooth:
            y1 = make_monotonic(x,y1)
            #y0 = make_monotonic(x,y0)
 
        if fluid == 'FV H+He':
            y1 = y1[::-1]
            x = x[::-1]
            y0 = y0[::-1]

        if sinusoidal_FV:
            ru = float(self.get('Gravity turns off between radii Low', fname=fname1))
            idxu = np.argmin(np.abs(x - ru))
            rl = float(self.get('Gravity turns on between radii High', fname=fname1))
            idxl = np.argmin(np.abs(x - rl))
        else:
            # restrict the computational domain to only where the regions are mixed
            idxu = np.where( y1 != 1. )[0][-1] + 1
            idxl = np.where( y1 != 0. )[0][0] - 1
        if not silent:
            print(idxl, idxu)
            print(y1)
        y1 = y1[idxl:idxu]
        y0 = y0[idxl:idxu]
        x = x[idxl:idxu]
        if not silent:
            print(x[0], x[-1])

        # now we want to exclude any zones where the abundances
        # of neighboring cells are the same. This is hopefully
        # rare inside the computational domain and limited to only
        # a very small number of zones
        indexarray = np.where(np.diff(y1) == 0)[0]
        if not silent:
            print('removing zones:', indexarray)
        y1 = np.delete(y1,indexarray)
        y0 = np.delete(y0,indexarray)
        x = np.delete(x,indexarray)
                
        dt = float(deltat)

        # Calculate D starting from outer boundary:
        D = np.zeros(len(x))
        m = len(x) - 1
        # now do the solution:
        for i in range(m,1,-1):
            xl = np.float64(x[i] - x[i-1])
            r = np.float64(y0[i] - y1[i])
            p = np.float64(dt * (y1[i] - y1[i-1]) / (xl * xl))
            if i == m:
                D[i] = np.float64(r / p)
            else:
                xr = np.float64(x[i+1] - x[i])
                xm = np.float64(xl + xr) / 2.
                q = np.float64(dt * (y1[i] - y1[i+1]) / (xr * xm))
                D[i] = np.float64((r - q * D[i+1]) / p)
        
                    
        D = D * 1.e16 # Mm^2/s ==> cm^2/s
        if not silent:
            print(D)
        x = x * 1e8   # Mm ==> cm

        cb = utils.colourblind
        lsty = utils.linestyle
        if showfig:
            pl.figure()
            yplot = np.log10(y0long) if log_X else y0long
            pl.plot(xlong,yplot,\
                    marker='o',
                    color=cb(8),\
                    markevery=lsty(1)[1],\
                    mec = cb(8),
                    mew = 1.,
                    mfc = 'w',
                    label='$X_{'+str(fname1)+'}$')
            yplot = np.log10(y1long) if log_X else y1long
            pl.plot(xlong,yplot,\
                    marker='o',\
                    color=cb(9),\
                    lw=0.5,
                    markevery=lsty(2)[1],\
                    label='$X_{'+str(fname2)+'}$')
            lbl = '$\log_{10}\,X$ ' if log_X else '$X$ '
            pl.ylabel(lbl)
            pl.xlabel('$\mathrm{r\,/\,Mm}$')
            if Xlim is not None:
                pl.ylim(Xlim)
            else:
               pl.ylim(-8,0.1)
            pl.legend(loc='center right').draw_frame(False)
            if grid:
                pl.grid()
            pl.twinx()
            pl.plot(x/1.e8,np.log10(D),'k-',\
                    label='$D$') #'$D > 0$')

            if plot_Dlt0:
                pl.plot(x/1.e8,np.log10(-D),'k--',\
                        label='$D < 0$')
            pl.xlim((3.5, 9.5))
            if Dlim is not None:
                pl.ylim(Dlim)
            else:
                pl.ylim((8., 18.))
            pl.ylabel('$\log_{10}(D\,/\,{\\rm cm}^2\,{\\rm s}^{-1})$')
            pl.legend(loc='upper right').draw_frame(False)

        if returnY:
            return x/1.e8, D, y0, y1
        else:
            return x/1.e8,D

    def Dsolvedownexp(self,fname1,fname2,fluid='FV H+He',numtype='ndump',newton=False,niter=3,
                   debug=False,grid=False,FVaverage=False,tauconv=None,returnY=False):
        '''
        Solve diffusion equation sequentially by iterating over the spatial
        domain inwards from the upper boundary. This version of the method is
        explicit.
        
        Parameters
        ----------
        fname1,fname2 : int or float
            cycles from which to take initial and final abundance profiles
            for the diffusion step we want to mimic.
        fluid : string
            Which fluid do you want to track?
        numtype : string, optional
            Designates how this function acts and how it interprets
            fname.  If numType is 'file', this function will get the
            desired attribute from that file.  If numType is 'NDump'
            function will look at the cycle with that nDump.  If
            numType is 'T' or 'time' function will find the _cycle
            with the closest time stamp.
            The default is "ndump".
        newton : boolean, optional
            Whether or not to apply Newton-Raphson refinement of the
            solution for D.
            The default is False
        niter : int, optional
            If N-R refinement is to be done, niter is how many iterations
            to compute.
            The default is 3.
        grid : boolean, optional
            whether or not to show the axes grids.
            The default is False.
        FVaverage : boolean, optional
            Whether or not to average the abundance profiles over a
            convective turnover timescale. See also tauconv.
            The default is False.
        tauconv : float, optional
            If averaging the abundance profiles over a convective turnover
            timescale, give the convective turnover timescale (seconds).
            The default value is None.
        returnY : boolean, optional
            If True, return abundance vectors as well as radius and diffusion
            coefficient vectors
            The default is False.
        
        Output
        ------
        x : array
            radial co-ordinates (Mm) for which we have a diffusion coefficient
        D : array
            Diffusion coefficient (cm^2/s)
        
        Example
        -------
        YY=ppm.yprofile(path_to_data)
        YY.Dsolvedown(1,640)
        '''
        
        
        xlong = self.get('Y',fname=fname1,resolution='l') # for plotting
        if debug: print(xlong)
        x = xlong
        #        x = x * 1.e8
        
        def mf(fluid,fname):
            '''
                Get mass fraction profile of fluid 'fluid' at fname.
                '''
            y = self.get(fluid,fname=fname,resolution='l')
            if fluid == 'FV H+He':
                rhofluid = self.get('Rho H+He',fname=fname,resolution='l')
            else:
                rhofluid = self.get('RHOconv',fname=fname,resolution='l')
            rho = self.get('Rho',fname=fname,resolution='l')
            y = rhofluid * y / rho
            return y
        
        if FVaverage is False:
            y1 = mf(fluid,fname2)
            y1long = y1 # for plotting
            
            y0 = mf(fluid,fname1)
            y0long = y0 # for plotting
        else:
            if tauconv is None:
                raise IOError("Please define tauconv")
            # Find the dumps accross which one should average:
            # first profile:
            myt0 = self.get('t',fname1)[-1]
            myt01 = myt0 - tauconv / 2.
            myt02 = myt0 + tauconv / 2.
            myidx01 = np.abs(self.get('t') - myt01).argmin()
            myidx02 = np.abs(self.get('t') - myt02).argmin()
            mycyc01 = self.cycles[myidx01]
            mycyc02 = self.cycles[myidx02]
            # second profile:
            myt1 = self.get('t',fname2)[-1]
            myt11 = myt1 - tauconv / 2.
            myt12 = myt1 + tauconv / 2.
            myidx11 = np.abs(self.get('t') - myt11).argmin()
            myidx12 = np.abs(self.get('t') - myt12).argmin()
            mycyc11 = self.cycles[myidx11]
            mycyc12 = self.cycles[myidx12]
            # do the average for the first profile:
            ytmp = np.zeros(len(x))
            count=0
            for cyc in range(mycyc01,mycyc02):
                ytmp += mf(fluid,cyc)
                count+=1
            y0 = ytmp / float(count)
            # do the average for the first profile:
            ytmp = np.zeros(len(x))
            count=0
            for cyc in range(mycyc11,mycyc12):
                ytmp += mf(fluid,cyc)
                count+=1
            y1 = ytmp / float(count)
            
            y0long = y0
            y1long = y1
        
        if fluid == 'FV H+He':
            y1 = y1[::-1]
            x = x[::-1]
            y0 = y0[::-1]
        
        if debug: print(len(xlong), len(y0long))
        
        idx0 = np.abs(np.array(self.cycles) - fname1).argmin()
        idx1 = np.abs(np.array(self.cycles) - fname2).argmin()
        t0 = self.get('t')[idx0]
        t1 = self.get('t')[idx1]
        deltat = t1 - t0
        
        # now we want to exclude any zones where the abundances
        # of neighboring cells are the same. This is hopefully
        # rare inside the computational domain and limited to only
        # a very small number of zones
        indexarray = np.where(np.diff(y0) == 0)[0]
        print('removing zones:', indexarray)
        y1 = np.delete(y1,indexarray)
        y0 = np.delete(y0,indexarray)
        x = np.delete(x,indexarray)
        
        dt = float(deltat)
        
        # Calculate D starting from outer boundary:
        D = np.zeros(len(x))
        m = len(x) - 1
        # now do the solution:
        for i in range(m,1,-1):
            xl = np.float64(x[i] - x[i-1])
            r = np.float64(y0[i] - y1[i])
            p = np.float64(dt * (y0[i] - y0[i-1]) / (xl * xl))
            if i == m:
                D[i] = np.float64(r / p)
            else:
                xr = np.float64(x[i+1] - x[i])
                xm = np.float64(xl + xr) / 2.
                q = np.float64(dt * (y0[i] - y0[i+1]) / (xr * xm))
                D[i] = np.float64((r - q * D[i+1]) / p)
        
        
        D = D * 1.e16 # Mm^2/s ==> cm^2/s
        x = x * 1e8   # Mm ==> cm
        
        pl.figure()
        pl.plot(xlong,np.log10(y0long),utils.linestyle(1)[0],\
                markevery=utils.linestyle(1)[1],\
                label='fluid above'+' '+str(fname1))
        pl.plot(xlong,np.log10(y1long),utils.linestyle(2)[0],\
                markevery=utils.linestyle(2)[1],\
                label='fluid above'+' '+str(fname2))
        pl.ylabel('$\log\,X$ '+fluid.replace('FV',''))
        pl.xlabel('r / Mm')
        pl.ylim(-8,0.1)
        pl.legend(loc='lower left').draw_frame(False)
        if grid:
            pl.grid()
        pl.twinx()
        pl.plot(x/1.e8,np.log10(D),'k-',\
                label='$D$') #'$D > 0$')
        pl.plot(x/1.e8,np.log10(-D),'k--',\
                label='$D < 0$')
        pl.ylabel('$\log D\,/\,{\\rm cm}^2\,{\\rm s}^{-1}$')
        pl.legend(loc='upper right').draw_frame(False)
        
        if returnY:
            return x/1.e8, D, y0, y1
        else:
            return x/1.e8,D
        
    def plot_entrainment_rates(self,dumps,r1,r2,fit=False,fit_bounds=None,save=False,lims=None,
                              Q = 1.944*1.60218e-6/1e43,RR = 8.3144598,amu = 1.66054e-24/1e27,
                              airmu = 1.39165,cldmu = 0.725,fkair = 0.203606102635,
                              fkcld = 0.885906040268,AtomicNoair = 6.65742024965,
                              AtomicNocld = 1.34228187919):
        '''
        Plots entrainment rates for burnt and unburnt material
        Parameters
        ----------
        data_path : string
            data path
        r1/r2 : float
            This function will only search for the convective 
            boundary in the range between r1/r2
        fit : boolean, optional
            show the fits used in finding the upper boundary
        fit_bounds : [int,int]
            The time to start and stop the fit for average entrainment
            rate units in minutes
        save : bool, optional
            save the plot or not
        lims : list, optional
            axes lims [xl,xu,yl,yu]

        '''
        atomicnocldinv = 1./AtomicNocld
        atomicnoairinv = 1./AtomicNoair
        patience0 = 5
        patience = 10

        nd = len(dumps)
        t = np.zeros(nd)
        L_H = np.zeros(nd)

        t00 = time.time()
        t0 = t00
        k = 0
        for i in range(nd):
            t[i] = self.get('t', fname = dumps[i], resolution = 'l')[-1]

            if dumps[i] >= 620:
                L_H[i] = self.get('L_C12pg', fname = dumps[i], resolution = 'l', airmu = airmu, \
                                   cldmu = cldmu, fkair = fkair, fkcld = fkcld, AtomicNoair = AtomicNoair, \
                                   AtomicNocld = AtomicNocld, corr_fact = 1.0)
            else:
                L_H[i] = 0.

            t_now = time.time()
            if (t_now - t0 >= patience) or \
               ((t_now - t00 < patience) and (t_now - t00 >= patience0) and (k == 0)):
                time_per_dump = (t_now - t00)/float(i + 1)
                time_remaining = (nd - i - 1)*time_per_dump
                print('Processing will be done in {:.0f} s.'.format(time_remaining))
                t0 = t_now
                k += 1

        ndot = L_H/Q
        X_H = fkcld*1./AtomicNocld
        mdot_L = 1.*amu*ndot/X_H
        dt = cdiff(t)
        m_HHe_burnt = (1e27/ast.msun_g)*np.cumsum(mdot_L*dt)        

        m_HHe_present = self.entrainment_rate(dumps,r1,r2, var='vxz', criterion='min_grad', offset=-1., \
                        integrate_both_fluids=False, show_output=False, return_time_series=True)

        m_HHe_total = m_HHe_present + m_HHe_burnt
        if fit_bounds is not None:
            idx2 = list(range(np.argmin(t/60. < fit_bounds[0]), np.argmin(t/60. < fit_bounds[1])))
            print(idx2)
            m_HHe_total_fc2 = np.polyfit(t[idx2], m_HHe_total[idx2], 1)
            m_HHe_total_fit2 = m_HHe_total_fc2[0]*t[idx2] + m_HHe_total_fc2[1]

            mdot2 = m_HHe_total_fc2[0]
            mdot2_str = '{:e}'.format(mdot2)
            parts = mdot2_str.split('e')
            mantissa = float(parts[0])
            exponent = int(parts[1])
            lbl2 = r'$\dot{{\mathrm{{M}}}}_\mathrm{{e}} = {:.2f} \times 10^{{{:d}}}$ M$_\odot$ s$^{{-1}}$'.\
                  format(mantissa, exponent)

        ifig = 1; pl.close(ifig); pl.figure(ifig)
        if fit:
            pl.plot(t[idx2]/60., m_HHe_total_fit2, '-', color =  'k', lw = 0.5, \
                     zorder = 102, label = lbl2)
        pl.plot(t/60., m_HHe_present, ':', color = cb(3), label = 'present')
        pl.plot(t/60., m_HHe_burnt, '--', color = cb(6), label = 'burnt')
        pl.plot(t/60., m_HHe_total, '-', color = cb(5), label = 'total')
        pl.xlabel('t / min')
        pl.ylabel(r'M$_\mathrm{HHe}$ [M_Sun]')
        if lims is not None:
            pl.axis(lims)
        pl.gca().ticklabel_format(style='sci', scilimits=(0,0), axis='y')
        pl.legend(loc = 0)
        pl.tight_layout()
        if save:
            pl.savefig('entrainment_rate.pdf')
            
    def entrainment_rate(self, cycles, r_min, r_max, var='vxz', criterion='min_grad', \
                         offset=0., integrate_both_fluids=False, 
                         integrate_upwards=False, show_output=True, ifig0=1, \
                         silent=True, mdot_curve_label=None, file_name=None,
                         return_time_series=False):
        
        def regrid(x, y, x_int):
            int_func = interpolate.CubicSpline(x[::-1], y[::-1])
            return int_func(x_int)
        
        r = self.get('Y', fname = cycles[0], resolution='l')
        
        idx_min = np.argmin(np.abs(r - r_max))
        idx_max = np.argmin(np.abs(r - r_min))

        r_min = r[idx_max]
        r_max = r[idx_min]
        
        r = r[idx_min:(idx_max + 1)]
        r_int = np.linspace(r_min, r_max, num = 20.*(idx_max - idx_min + 1))
        dr_int = cdiff(r_int)
        
        time = np.zeros(len(cycles))
        r_b = np.zeros(len(cycles))
        r_top = np.zeros(len(cycles))
        for i in range(len(cycles)):
            time[i] = self.get('t', fname = cycles[i], resolution='l')[-1]
            
            if var == 'vxz':
                q = self.get('EkXZ', fname = cycles[i], resolution='l')[idx_min:(idx_max + 1)]**0.5
            else:
                q = self.get(var, fname = cycles[i], resolution='l')[idx_min:(idx_max + 1)]
            
            q_int = regrid(r, q, r_int)
            grad = cdiff(q_int)/dr_int

            if criterion == 'min_grad':
                idx_b = np.argmin(grad)
            elif criterion == 'max_grad':
                idx_b = np.argmax(grad)
            else:
                idx_b = np.argmax(np.abs(grad))

            r_b[i] = r_int[idx_b]
            r_top[i] = r_b[i]
            
            # Optionally offset the integration limit by a multiple of q's
            # scale height.
            if np.abs(grad[idx_b]) > 0.:
                H_b = q_int[idx_b]/np.abs(grad[idx_b])
                r_top[i] += offset*H_b
        
        timelong = time
        delta = 0.05*(np.max(time) - np.min(time))
        timelong = np.insert(timelong,0, timelong[0] - delta)
        timelong = np.append(timelong, timelong[-1] + delta)
        
        # fc = fit coefficients
        r_b_fc = np.polyfit(time, r_b, 1)
        r_b_fit = r_b_fc[0]*timelong + r_b_fc[1]
        r_top_fc = np.polyfit(time, r_top, 1)
        r_top_fit = r_top_fc[0]*timelong + r_top_fc[1]
        
        m_ir = np.zeros(len(cycles))
        r = self.get('Y', fname = cycles[0], resolution='l')
        r_int = np.linspace(np.min(r), np.max(r), num = 20.*len(r))
        dr_int = cdiff(r_int)
        for i in range(len(cycles)):
            if integrate_both_fluids:
                rho = self.get('Rho', fname = cycles[i], resolution='l')
            else:
                rho_HHe = self.get('Rho H+He', fname = cycles[i], resolution='l')
                FV_HHe = self.get('FV H+He', fname = cycles[i], resolution='l')
                rho = rho_HHe*FV_HHe
            
            rho_int = regrid(r, rho, r_int)
            
            idx_top = np.argmin(np.abs(r_int - r_top[i]))
            dm = 4.*np.pi*r_int**2*dr_int*rho_int
            
            if integrate_upwards:
                m_ir[i] = np.sum(dm[(idx_top + 1):-1])
            else:
                m_ir[i] = np.sum(dm[0:(idx_top + 1)])
        
        # fc = fit coefficients
        m_ir *= 1e27/ast.msun_g
        m_ir_fc = np.polyfit(time, m_ir, 1)
        m_ir_fit = m_ir_fc[0]*timelong + m_ir_fc[1]
        if integrate_upwards:
            mdot = -m_ir_fc[0]
        else:
            mdot = m_ir_fc[0]

        if show_output:
            cb = utils.colourblind
            pl.close(ifig0); fig1 = pl.figure(ifig0)
            pl.plot(time/60., r_top, color = cb(5), ls = '-', label = r'r$_\mathrm{top}$')
            pl.plot(time/60., r_b, color = cb(8), ls = '--', label = r'r$_\mathrm{b}$')
            pl.plot(timelong/60., r_top_fit, color = cb(4), ls = '-', lw = 0.5)
            pl.plot(timelong/60., r_b_fit, color = cb(4), ls = '-', lw = 0.5)
            pl.xlabel('t / min')
            pl.ylabel('r / Mm')
            xfmt = ScalarFormatter(useMathText = True)
            pl.gca().xaxis.set_major_formatter(xfmt)
            pl.legend(loc = 0)
            fig1.tight_layout()

            if not silent:
                print('r_b is the radius of the convective boundary.')
                print('r_b_fc = ', r_b_fc)
                print('dr_b/dt = {:.2e} km/s\n'.format(1e3*r_b_fc[0]))
                print('r_top is the upper limit for mass integration.')
                print('dr_top/dt = {:.2e} km/s'.format(1e3*r_top_fc[0]))
            
            max_val = np.max(m_ir)
            #if show_fits:
            max_val = np.max((max_val, np.max(m_ir_fit)))
            max_val *= 1.1 # allow for some margin at the top
            oom = int(np.floor(np.log10(max_val)))
            
            pl.close(ifig0 + 1); fig2 = pl.figure(ifig0 + 1)
            pl.plot(time/60., m_ir/10**oom, color = cb(5), label=mdot_curve_label)
            mdot_str = '{:e}'.format(mdot)
            parts = mdot_str.split('e')
            mantissa = float(parts[0])
            exponent = int(parts[1])
            #if show_fits:
            if integrate_upwards:
                lbl = r'$\dot{{\mathrm{{M}}}}_\mathrm{{a}} = {:.2f} \times 10^{{{:d}}}$ M$_\odot$ s$^{{-1}}$'.\
                          format(-mantissa, exponent)
            else:
                lbl = r'$\dot{{\mathrm{{M}}}}_\mathrm{{e}} = {:.2f} \times 10^{{{:d}}}$ M$_\odot$ s$^{{-1}}$'.\
                          format(mantissa, exponent)
            pl.plot(timelong/60., m_ir_fit/10**oom, color = cb(4), ls = '-', lw = 0.5, label = lbl)
            pl.xlabel('t / min')
            if integrate_upwards:
                sub = 'a'
            else:
                sub = 'e'
            ylbl = r'M$_{:s}$ / 10$^{{{:d}}}$ M$_\odot$'.format(sub, oom)
            if oom == 0.:
                ylbl = r'M$_{:s}$ / M$_\odot$'.format(sub)
                
            pl.ylabel(ylbl)
            yfmt = FormatStrFormatter('%.1f')
            fig2.gca().yaxis.set_major_formatter(yfmt)
            fig2.tight_layout()
            if integrate_upwards:
                loc = 1
            else:
                loc = 2
            pl.legend(loc = loc)
            if file_name is not None:
                fig2.savefig(file_name)
        
            if not silent:
                print('Resolution: {:d}^3'.format(2*len(r)))
                print('m_ir_fc = ', m_ir_fc)
                print('Entrainment rate: {:.3e} M_Sun/s'.format(mdot))
        
        if return_time_series:
            return m_ir
        else:
            return mdot

    def boundary_radius(self, cycles, r_min, r_max, var='vxz', \
                        criterion='min_grad', var_value=None):
        eps = 1e-9
        n_cycles = len(cycles)
        rb = np.zeros(n_cycles)

        r = self.get('Y', cycles[0], resolution='l')
        idx_r_min = np.argmin(np.abs(r - r_min))
        idx_r_max = np.argmin(np.abs(r - r_max))

        for i in range(n_cycles):
            if var == 'vxz':
                v = self.get('EkXZ', fname = cycles[i], resolution='l')**0.5
            else:
               v = self.get(var, cycles[i], resolution='l')

            if criterion == 'min_grad' or criterion == 'max_grad':
                # The following code always looks for a local minimum in dv.
                dvdr = cdiff(v)/cdiff(r)

                # A local maximum is found by looking for a local minimum in
                # -dvdr.
                if criterion == 'max_grad':
                    dvdr = -dvdr

                # 0th-order estimate.
                idx0 = idx_r_max + np.argmin(dvdr[idx_r_max:idx_r_min])
                r0 = r[idx0]

                # Try to pinpoint the radius of the local minimum by fitting
                # a parabola around r0.
                coefs = np.polyfit(r[idx0-1:idx0+2], dvdr[idx0-1:idx0+2], 2)

                if np.abs(coefs[0]) > eps:
                    r00 = -coefs[1]/(2.*coefs[0])

                    # Only use the refined radius if it is within the three
                    # cells.
                    if r00 < r[idx0-1] and r00 > r[idx0+1]:
                        r0 = r00
            elif criterion == 'value':
                # 0th-order estimate.
                idx0 = idx_r_max + np.argmin(np.abs(v[idx_r_max:idx_r_min] - var_value))
                r0 = r[idx0]

                if np.abs(v[idx0] - var_value) > eps:
                    # 1st-order refinement.
                    if idx0 > idx_r_max and idx0 < idx_r_min:
                        if (v[idx0-1] < var_value and v[idx0] > var_value) or \
                           (v[idx0-1] > var_value and v[idx0] < var_value):
                            slope = v[idx0] - v[idx0-1]
                            t = (var_value - v[idx0-1])/slope
                            r0 = (1. - t)*r[idx0-1] + t*r[idx0]
                        elif (v[idx0] < var_value and v[idx0+1] > var_value) or \
                            (v[idx0] > var_value and v[idx0+1] < var_value):
                            slope = v[idx0+1] - v[idx0]
                            t = (var_value - v[idx0])/slope
                            r0 = (1. - t)*r[idx0] + t*r[idx0+1]
                        else:
                            r0 = r_max
                        
            rb[i] = r0

        return rb

    def vaverage(self,vi='v',transient=0.,sparse=1,showfig = False):
        '''
        plots and returns the average velocity profile for a given
        orientation (total, radial or tangential) over a range of dumps
        and excluding an initial user-specified transient in seconds.
        '''
            
        cycs = self.cycles
        time = self.get('t')
        istart = np.abs(time-transient).argmin()
        cycs = cycs[istart::sparse]

        Y    = self.get('Y',fname=1,resolution='l')
        
        if vi == 'v':
            Ei ='Ek'
            ylab='$\log~v_\mathrm{tot}$'
        if vi == 'vY':
            Ei = 'EkY'
            ylab='$\log~v_\mathrm{Y}$'
        if vi == 'vXZ':
            Ei = 'EkXZ'
            ylab='$\log~v_\mathrm{XZ}$'

        vav = np.zeros(len(Y))
        
        for cyc in cycs:
            Ek   = self.get(Ei,fname=cyc,resolution='l')
            if vi == 'v':
                v    = np.sqrt(2.*array(Ek,dtype=float))
            else:
                v    = np.sqrt(array(Ek,dtype=float))

            vav += v
            
        vav = vav * 1.e8 / len(cycs) # average in cm / s
        
        if showfig:
            pl.figure()
            pl.plot(Y,np.log10(vav),'r-')
            pl.ylabel(ylab)
            pl.xlabel('r / Mm')

        return vav

# below are some utilities that the user typically never calls directly
    def readTop(self,atri,filename,stddir='./'):
        """ 
        Private routine that Finds and returns the associated value for
        attribute in the header section of the file.
        
        Input:
        atri, what we are looking for.
        filename where we are looking.
        StdDir the directory where we are looking, Defaults to the
        working Directory.

        """
        if stddir.endswith('/'):
            filename = str(stddir)+str(filename)
        else:
            filename = str(stddir)+'/'+str(filename)
        f=open(filename,'r')
        headerLines=[]
        header=[]
        headerAttri=[]
        for i in range(0,10): # Read out the header section of the file.
            line = f.readline()
            line=line.strip()
            if line != '':
                headerLines.append(line)
        f.close()
        for i in range(len(headerLines)): #for each line of header data split up the occurances of '    '
            header.extend(headerLines[i].split('     '))
            header[i]=header[i].strip()

        for i in range(len(header)):
            tmp=header[i].split('=')# for each line split up on occurances of =
            if len(tmp)!=2: # If there are not two parts, add the unsplit line to headerAttri
                tmp=[]
                tmp.append(header[i].strip())
                headerAttri.append(tmp)
            elif len(tmp)==2: # If there are two parts, add the list of the two parts to headerAttri
                tmp[0]=tmp[0].strip()
                tmp[1]=tmp[1].strip()
                headerAttri.append([tmp[0],tmp[1]])

        for i in range(len(headerAttri)):
            if atri in headerAttri[i]: # if the header arrtibute equals atri, return its associated value
                value=headerAttri[i][1]

        value =value.partition(' ')
        value=value[0]
        return value

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

    def spacetime_diagram(self, var_name, nt, fig, tlim=None, rlim=None, vlim=None, logscale=True, \
                  cmap='viridis', aspect=1./3., zero_intervals=None, patience0 = 5, patience = 30, \
                  **kwargs):
        if var_name == 'Ek':
            cbar_lbl = r'e$_\mathrm{k}$ / erg g$^{-1}$'
            unit = 1e43/1e27
        elif var_name == 'enuc_C12pg':
            cbar_lbl = r'$\epsilon_\mathrm{C12pg}$ / erg cm$^{-3}$ s$^{-1}$'
            unit = 1e43/1e24
        elif var_name == 'enuc_C12C12':
            cbar_lbl = r'$\epsilon_\mathrm{C12C12}$ / erg cm$^{-3}$ s$^{-1}$'
            unit = 1e43/1e24
        else:
            cbar_lbl = var_name
            unit = 1.
        
        r = self.get('Y', fname = 0, resolution = 'l')
        if rlim is None:
            rlim = [r[-1], r[0]]
            
        ridx0 = np.argmin(np.abs(r - rlim[1]))
        ridx1 = np.argmin(np.abs(r - rlim[0]))
        nr = ridx1-ridx0+1
        ridx = np.linspace(ridx0, ridx1, num=nr, dtype=np.int32)
        
        r = r[ridx]

        if tlim is None:
            tlim = [0., 0.]
            tlim[0] = self.get('t', silent = True)[0]
            tlim[1] = self.get('t', silent = True)[-1]
        t = np.linspace(tlim[0], tlim[1], nt)

        zero = np.zeros(nt, dtype = bool)
        if zero_intervals is not None:
            for i in range(len(zero_intervals)/2):
                idx = where((t >= zero_intervals[2*i]) & \
                            (t <= zero_intervals[2*i + 1]))
                zero[idx] = True
        
        var = np.zeros((nr, nt))

        t00 = time.time()
        t0 = t00
        n = 0
        k = 0
        n_nonzero = nt - np.count_nonzero(zero)
        for i in range(nt):
            if zero[i]:
                continue
            
            var[:, i] = unit*self.get(var_name, fname = t[i], numtype = 'time', \
                                      resolution = 'l', silent = True, **kwargs)[ridx]
            
            n += 1
            t_now = time.time()
            if (t_now - t0 >= patience) or \
               ((t_now - t00 < patience) and (t_now - t00 >= patience0) and (k == 0)):
                time_per_dump = (t_now - t00)/float(n)
                time_remaining = (n_nonzero - n - 1.)*time_per_dump
                print('Processing will be done in {:.0f} s.'.format(time_remaining))
                t0 = t_now
                k += 1

        if vlim is None:
            if logscale:
                vlim = [np.min(var[where(var > 0)]), \
                        np.max(var[where(var > 0)])]
            else:
                vlim = [np.min(var), np.max(var)]
                
            print('vlim = [{:.3e}, {:.3e}]'.format(vlim[0], vlim[1]))
        
        var[where(var < vlim[0])] = vlim[0]
        var[where(var > vlim[1])] = vlim[1]
        ax1 = fig.add_subplot(111)
            
        extent = (t[0]/60., t[-1]/60., r[-1], r[0])
        aspect *= (extent[1] - extent[0])/(extent[3] - extent[2])
        
        if logscale:
            norm = colors.LogNorm(vmin=vlim[0], vmax=vlim[1], clip=True)
        else:
            norm = colors.Normalize(vmin=vlim[0], vmax=vlim[1], clip=True)
        
        ax1i = ax1.imshow(var, aspect = aspect, cmap = cmap, extent = extent, \
                        norm=norm, interpolation = 'spline16')
        ax1.get_yaxis().set_tick_params(which='both', direction='out')
        ax1.get_xaxis().set_tick_params(which='both', direction='out')
        ax1.set_xlabel('t / min')
        ax1.set_ylabel('r / Mm')
        
        cbar = fig.colorbar(ax1i, orientation='vertical')
        cbar.set_label(cbar_lbl)


##########################################################
# mapping of visualisation variables from Robert Andrassy:
##########################################################

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
        print('first/last colour indices:', indices_normed[-1])
        print('correcting to 1.')
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
        lut = ppm.LUT('./ /BW-1536-UR-3.lut', s0=5., s1=245.499,
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
            print('ticks out of range')
            return

        print('min/max ticks being set to:', minval, maxval)
        print('corresponding to colour indices:', minidx, maxidx)
        print('ticks being placed at:', ticks)
        print('with colour indices:', colour_index_ticks)

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

        print(ileft, iright, minidx, maxidx)

        to_splice = copy.deepcopy(colours)[ileft:iright]
        newcolours = [[minidx, (rl, gl, bl)]] + to_splice + [[maxidx, (rr, gr, br)]]

        # now normalise the indices to [0-255]
        indices = np.array([c[0] for c in newcolours])
        newindices = 255.*(indices - np.min(indices)) / indices.ptp()
        # renormalise index tick locations as well
        colour_index_ticks = 255.*\
            (colour_index_ticks - np.min(colour_index_ticks)) / colour_index_ticks.ptp()
        print('new colour indices:', newindices)
        print('ticks now at:', colour_index_ticks)

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
        cbar.ax.tick_params(axis='both', which='both',length=0,labelsize=6)
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

def upper_bound_ut(data_path, dump_to_plot, hist_dump_min,
                   hist_dump_max, ylims = None, derivative = False, r1=7.4, r2=8.4, silent = True):

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
    nr = len(rp.get('y'))
    
    sparse = 1
    dumps = np.array([rp_set.dumps[i] for i in range(0, len(rp_set.dumps), sparse)])

    n_dumps = len(dumps)
    n_buckets = rp_set.get_dump(rp_set.dumps[0]).get('nbuckets')
    t = np.zeros(n_dumps)
    r_ub = np.zeros((n_buckets, n_dumps))
    ut = np.zeros((nr, n_buckets, n_dumps))
    dutdr = np.zeros((nr, n_buckets, n_dumps))

    for k in range(n_dumps):
        rp = rp_set.get_dump(rp_set.dumps[k])
        t[k] = rp.get('time')

        res = analyse_dump(rp, r1, r2)
        r = res[0]
        ut[:, :, k] = res[1]
        dutdr[:, :, k] = res[2]
        r_ub[:, k] = res[3]

    avg_r_ub = np.sum(r_ub, axis = 0)/float(n_buckets)
    dev = np.array([r_ub[i, :] - avg_r_ub for i in range(n_buckets)])
    sigmap_r_ub = np.zeros(n_dumps)
    sigmam_r_ub = np.zeros(n_dumps)
    idx = np.argmin(np.abs(dumps - dump_to_plot))
    
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
    if not silent:
        print("Dump {:d} (t = {:.2f} min).".format(dump_to_plot, t[dump_to_plot]/60.))
        print("Histogram constructed using dumps {:d} (t = {:.2f} min) to {:d} (t = {:.2f} min) inclusive."\
            .format(hist_dump_min, t[hist_dump_min]/60., hist_dump_max, t[hist_dump_max]/60.))

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

        ax0.plot(r, temp[:, bucket, idx], ls = '-', lw = 0.5, color = cb(3), \
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
    if ylims is not None:
        ax0.set_ylim(ylims)
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
    
def get_avg_rms_velocities(prof, dumps, comp):
    '''
    Finds an average velocity vector as a function of radius for a given
    range of dumps. To find a velocity vector as a function of radius see
    get_v_evolution().
    
    Parameters
    ----------
    yprof : yprofile instance
        profile to examine
    dumps : range
        dumps to average over
    comp : string
        component to use 'r':'radial','t':'tangential','tot':'total'
    '''
    avg_rms_v = 0.
    
    for d in dumps:
        if comp == 'tot':
            rms_v = np.sqrt(2.*prof.get('Ek', fname=d, resolution='l'))
        elif comp == 'r':
            rms_v = np.sqrt(prof.get('EkY', fname=d, resolution='l'))
        elif comp == 't':
            rms_v = np.sqrt(prof.get('EkXZ', fname=d, resolution='l'))
        
        avg_rms_v += rms_v
        
    avg_rms_v /= float(len(dumps))

    return avg_rms_v
    
def get_v_evolution(prof, cycles, r1, r2, comp, RMS):
    '''
    Finds a velocity vector and a time vector for a range of cycles
    
    Parameters
    ----------
    prof : yprofile object
        prof to look at
    cycles : range
        cycles to look at
    r1/r1 : float
        boundaries of the range to look for v max in
    comp : string
        velocity component to look for 'radial' or 'tangential'
    RMS : string
        'mean' 'min' or 'max velocity
    '''
        
    r = prof.get('Y', fname = cycles[0], resolution = 'l')
    idx1 = np.argmin(np.abs(r - r1))
    idx2 = np.argmin(np.abs(r - r2))

    t = np.zeros(len(cycles))
    v = np.zeros(len(cycles))
    for k in range(len(cycles)):
        t[k] = prof.get('t', fname = cycles[k], resolution = 'l')[-1]
        if comp == 'radial':
            v_rms  = prof.get('EkY', fname = cycles[k], resolution = 'l')**0.5
        elif comp == 'tangential':
            v_rms  = prof.get('EkXZ', fname = cycles[k], resolution = 'l')**0.5
        elif comp == 'tot':
            v_rms = (2.*prof.get('Ek', fname = cycles[k], resolution = 'l'))**0.5
        if RMS == 'mean':
            v[k] = np.mean(v_rms[idx2:idx1])
        elif RMS == 'min':
            v[k] = np.min(v_rms[idx2:idx1])
        elif RMS == 'max':
            v[k] = np.max(v_rms[idx2:idx1])                
    return t, v

def v_evolution(cases, ymin, ymax, comp, RMS, sparse = 1, markevery = 25, ifig = 12, first_dump = 0,
               last_dump = -1, lims = None):

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
    comp : string
        component that the velocity is in. option: 'radial', 'tangential'
    RMS : string
        options: 'min', 'max' or 'mean'. What aspect of the RMS velocity
        to look at
    first_dump,last_dump : int, optional
        Gives the range of dumps to plot if larger than last dump
        will plot up to last dump
    lims : array
        axes lims [xl,xu,yl,yu]
    
    Example
    -------
    
    ppm.set_YProf_path('/data/ppm_rpod2/YProfiles/O-shell-M25/',YProf_fname='YProfile-01-0000.bobaaa')
    v_evolution(['D15','D2','D1'], 4., 8.,'max','radial')

    '''
    pl.close(ifig),pl.figure(ifig)
    cb = utils.colourblind
    ls = utils.linestylecb
    yy = 0
    for case in cases:

        try:
            prof = yprofile(ppm_path+case)
        except ValueError:
            print("have you set the yprofile filepath using ppm.set_YProf_path?")
        if last_dump > prof.cycles[-1]:
            end = -1
        else:
            end = last_dump
        cycles = list(range(prof.cycles[first_dump], prof.cycles[end], sparse))
        t, vr_max = get_v_evolution(prof, cycles, ymin, ymax, comp, RMS)
        pl.plot(t/60.,  1e3*vr_max,  color = cb(yy),\
                 marker = ls(yy)[1], markevery = markevery, label = case)
        yy += 1

    pl.xlabel('t / min')
    pl.ylabel(r'v$_r$ / km s$^{-1}$')
    if lims is not None:
        pl.axis(lims)
        print('p')
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
        dumps = list(range(rp_set.dumps[0],\
                      #rp_set.dumps[0]+100,1)
                      rp_set.dumps[-1]+1,1))
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
                                  cldmu = cldmu, fkair = fkair, fkcld = fkcld, AtomicNoair = AtomicNoair,
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
            print('Processing will be done in {:.0f} s.'.format(time_remaining))
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

def L_H_L_He_comparison(cases, sparse = 1, ifig=101, airmu = 1.39165, cldmu = 0.725,
    fkair = 0.203606102635,fkcld = 0.885906040268,AtomicNoair = 6.65742024965,
    AtomicNocld = 1.34228187919,markevery = 1,lims = None,save= False):
    
    '''
    Compares L_H to L_He, optional values are set to values from O-shell
    burning paper.
    
    Parameters
    ----------
    cases : string array
        names of yprofile instances i.e['D1','D2'...]
    sparse :int
        what interval in the range to calculate
        1 calculates every dump 2 every second dump ect
    lims : array
        plot limits
    save : boolean
        save figure
    '''
    yprofs = {}
    res = {}
    
    for case in cases:
        
        try:
            yprofs[case] = yprofile(ppm_path+case)
        except ValueError:
            print("have you set the yprofile filepath using ppm.set_YProf_path?")
        
        r = yprofs[case].get('Y', fname=0, resolution='l')
        res[case] = 2*len(r)

    patience0 = 5
    patience = 60

    dumps = {}
    nd = {}
    t = {}
    L_H = {}
    L_He = 2.25*2.98384E-03

    for this_case in cases:
        print('Processing {:s}...'.format(this_case))
        
        dumps[this_case] = np.arange(min(yprofs[this_case].ndumpDict.keys()),\
           max(yprofs[this_case].ndumpDict.keys()) + 1, sparse)
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
            L_H[this_case][i] = yprofs[this_case].get('L_C12pg', fname = dumps[this_case][i],
                                resolution = 'l', airmu = airmu, cldmu = cldmu,
                                fkair = fkair, fkcld = fkcld,  AtomicNoair = AtomicNoair,
                                AtomicNocld = AtomicNocld, corr_fact = 1.5)

            t_now = time.time()
            if (t_now - t0 >= patience) or \
               ((t_now - t00 < patience) and (t_now - t00 >= patience0) and (k == 0)):
                time_per_dump = (t_now - t00)/float(i + 1)
                time_remaining = (nd[this_case] - i - 1)*time_per_dump
                print('Processing will be done in {:.0f} s.'.format(time_remaining))
                t0 = t_now
                k += 1

    pl.close(ifig); pl.figure(ifig)
    pl.axhline((1e43/ast.lsun_erg_s)*L_He, ls = '--', color = cb(4), \
        label = r'L$_\mathrm{He}$')
    i =0
    for this_case in cases:
        lbl = r'{:s} $\left({:d}^3\right)$'.format(this_case, res[this_case])
        pl.semilogy(t[this_case]/60., (1e43/ast.lsun_erg_s)*L_H[this_case], \
            ls = '-', color = cb(i), marker= 's', \
            label = this_case)
        i+=1
    if lims is not None:
        pl.axis(lims)
    pl.xlabel('t / min')
    pl.ylabel(r'L$_H$ / L$_\odot$')
    pl.legend(loc=0, ncol=2)
    pl.tight_layout()
    if save:
        pl.savefig('L_H-L_He_'+cases[0]+cases[1]+cases[2]+'.pdf')

def get_upper_bound(data_path, r1, r2):

    '''
    Returns information about the upper convective boundary
    
    Parameters
    ----------
    r1/r2 = float
        This function will only search for the convective 
        boundary in the range between r1/r2
    
    Returns
    ------
    [all arrays]
    avg_r_ub : average radius of upper boundary
    sigmam_r_ub/sigmap_r_ub: 2 \sigma fluctuations in upper boundary
    r_ub : upper boundary
    t: time
    
    '''

    rp_set = rprof.rprofile_set(data_path)

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

def get_r_int(data_path, r_ref, gamma, sparse = 1): 
    '''
    Returns values for radius of interface for a given 
    reference radius.
    
    Parameters
    ----------
    data_path : string
        data path
    r_ref : float
        r_int is the radius of the interface at which the relative 
        [A(t) - A0(t)]/A0(t) with respect to a reference value A0(t)
        exceeds a threshold. The reference value A0(t) is taken to be
        the average A at a reference radius r_ref at time t. 
        We need to put r_ref to the bottom of the convection zone,
        because most of the convection zone is heated in the 
        final explosion. The threshold should be a few times 
        larger than the typical bucket-to-bucket fluctuations 
        in A due to convection. A value of 1e-3 is a good trade-off
        between sensitivity and the amount of noise. 
    sparse: int, optional
        What interval to have between data points
        
    Returns
    ------
    [all arrays]
    avg_r_int : average radius of interface
    sigmam_r_int/sigmap_r_int: 2 \sigma fluctuations in r_int
    r_int : interface radius
    t: time
    '''
    rp_set = rprof.rprofile_set(data_path)
    rp = rp_set.get_dump(rp_set.dumps[0])
    n_buckets = rp.get('nbuckets')
    r = rp.get('y')
    idx_ref = np.argmin(np.abs(r - r_ref))

    dumps = np.arange(1, 1960, sparse)
    nd = len(dumps)

    dr = r[1] - r[0]
    r_int = np.zeros((n_buckets, nd))
    avg_r_int = np.zeros(nd)
    sigmap_r_int = np.zeros(nd)
    sigmam_r_int = np.zeros(nd)
    t = np.zeros(nd)
    
    for i in range(nd):
        rp = rp_set.get_dump(dumps[i])
        p = rp.get_table('p')[0, :, :]
        rho = rp.get_table('rho')[0, :, :]
        A = p/rho**gamma
        A0 = A[idx_ref, 0]
        t[i] = rp.get('time')

        for bucket in range(1, n_buckets + 1):
            rel_diff = (A[:, bucket] - A0)/A0

            threshold = 1e-3
            # 0-th order estimate first
            idx_top = np.argmax(rel_diff > threshold)
            r_int[bucket - 1, i] = r[idx_top]

            # refine to 1-st order now
            slope = (rel_diff[idx_top] - rel_diff[idx_top - 1])/dr
            r_int[bucket - 1, i] -= (rel_diff[idx_top] - threshold)/slope

        avg_r_int[i] = np.sum(r_int[:,i])/float(n_buckets)
        dev = np.array([r_int[b,i] - avg_r_int[i] for b in range(n_buckets)])
        devp = dev[dev >= 0]
        if len(devp) > 0:
            sigmap_r_int[i] = (sum(devp**2)/float(len(devp)))**0.5
        else:
            sigmap_r_int[i] = None

        devm = dev[dev <= 0]
        if len(devm) > 0:
            sigmam_r_int[i] = (sum(devm**2)/float(len(devm)))**0.5
        else:
            sigmam_r_int[i] = None
            
    return avg_r_int, sigmam_r_int, sigmap_r_int, r_int, t

def plot_boundary_evolution(data_path, r1, r2, t_fit_start=700,
                            silent = True, show_fits = False, ifig = 5,
                            r_int = False, r_ref = None, gamma = None,
                            sparse = 10, lims = None, insert = False):
    
    '''

    Displays the time evolution of the convective boundary or Interface radius

    Plots Fig. 14 or 15 in paper: "Idealized hydrodynamic simulations
    of turbulent oxygen-burning shell convection in 4 geometry"
    by Jones, S.; Andrassy, R.; Sandalski, S.; Davis, A.; Woodward, P.; Herwig, F.
    NASA ADS: http://adsabs.harvard.edu/abs/2017MNRAS.465.2991J

    Parameters
    ----------
    data_path : string
        data path
    r1/r2 : float
        This function will only search for the convective 
        boundary in the range between r1/r2
    show_fits : boolean, optional
        show the fits used in finding the upper boundary
    t_fit_start : int
        The time to start the fit for upper boundary fit takes 
        range t[t_fit_start:-1] and computes average boundary
    r_int : bool, optional
        True plots interface radius, False plots convective boundary
        !If true r_ref must have a value
    r_ref : float
        r_int is the radius of the interface at which the relative 
        [A(t) - A0(t)]/A0(t) with respect to a reference value A0(t)
        exceeds a threshold. The reference value A0(t) is taken to be
        the average A at a reference radius r_ref at time t. 
        We need to put r_ref to the bottom of the convection zone,
        because most of the convection zone is heated in the 
        final explosion. The threshold should be a few times 
        larger than the typical bucket-to-bucket fluctuations 
        in A due to convection. A value of 1e-3 is a good trade-off
        between sensitivity and the amount of noise. 
    sparse: int, optional
        What interval to have between data points
    lims : array
        limits of the insert axes [xl,xu,yl,yu]
    insert: bool
        whether or not to include a second inserted subplot
     
    '''
    cb = utils.colourblind
    
    rp_set = rprof.rprofile_set(data_path)

    n_dumps = len(rp_set.dumps)
    n_buckets = rp_set.get_dump(rp_set.dumps[0]).get('nbuckets')
    t = np.zeros(n_dumps)
    r_ub = np.zeros((n_buckets, n_dumps))
    
    if not r_int:
        avg_r_ub, sigmam_r_ub, sigmap_r_ub, r_ub, t = get_upper_bound(data_path, r1, r2)
    else:
        avg_r_ub, sigmam_r_ub, sigmap_r_ub, r_ub, t = get_r_int(data_path, r_ref, gamma, sparse = sparse)
    
    if show_fits:
        idx_fit_start = np.argmin(np.abs(t - t_fit_start))
        t_fit_start = t[idx_fit_start]

        # fc = fit coefficients
        fc_avg = np.polyfit(t[idx_fit_start:-1], avg_r_ub[idx_fit_start:-1], 1)
        avg_fit = fc_avg[0]*t + fc_avg[1]
        fc_plus = np.polyfit(t[idx_fit_start:-1], 2.*sigmap_r_ub[idx_fit_start:-1], 1)
        plus_fit = fc_plus[0]*t + fc_plus[1]
        fc_minus = np.polyfit(t[idx_fit_start:-1], 2.*sigmam_r_ub[idx_fit_start:-1], 1)
        minus_fit = fc_minus[0]*t + fc_minus[1]
    if not insert:
        pl.close(ifig); fig = pl.figure(ifig)#, figsize = (6.0, 4.7))
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
        pl.xlabel('t / min')
        if r_int:
            pl.ylabel(r'r$_\mathrm{int}$ / Mm')
        else:
            pl.ylabel(r'r$_\mathrm{ub}$ / Mm')
        pl.legend(loc = 0, frameon = False)
        if not silent:
            print('The fitting starts at t = {:.1f} s = {:.1f} min.'.format(t_fit_start, t_fit_start/60.))
            print('')
            print('Average:')
            print('{:.3e} Mm + ({:.3e} Mm/s)*t'.format(fc_avg[1], fc_avg[0]))
            print('')
            print('Positive fluctuations:')
            print('{:.3e} Mm + ({:.3e} Mm/s)*t'.format(fc_plus[1], fc_plus[0]))
            print('')
            print('Negative fluctuations:')
            print('{:.3e} Mm + ({:.3e} Mm/s)*t'.format(fc_minus[1], fc_minus[0]))
    else:
        def plot_data(ax):
            for i in range(n_buckets):
                lbl=''
                if i == 0:
                    lbl='bucket data'
                ax.plot(t/60., r_ub[i,:], '-', lw=0.5, color=cb(3), label=lbl)

            ax.plot(t/60., avg_r_ub, '-', lw=1.0, color=cb(4), label='mean')
            ax.plot(t/60., avg_r_ub + 2*sigmap_r_ub, '--', lw=1., color=cb(4), \
                     label = r'2$\sigma$ fluctuations')
            ax.plot(t/60., avg_r_ub - 2*sigmam_r_ub, '--', lw=1., color=cb(4))

        ifig = ifig; pl.close(ifig); fig = pl.figure(ifig)
        ax1 = fig.add_subplot(111)
        plot_data(ax1)
        ax1.set_xlim((0., t[-1]/60))
        ax1.set_ylim((12., 30.))
        ax1.set_xlabel('t / min')
        ax1.set_ylabel(r'r$_\mathrm{int}$ / Mm')
        ax1.legend(loc=0)

        left, bottom, width, height = [0.29, 0.44, 0.35, 0.35]
        ax2 = fig.add_axes([left, bottom, width, height])
        plot_data(ax2)
        ax2.set_xlim((lims[0], lims[1]))
        ax2.set_ylim((lims[2], lims[3]))
        fig.tight_layout()
        
def compare_entrained_material(yps, labels, fname, ifig = 1):
    
    '''
    Compares the entrainment rate of two seperate yprofile objects
    
    Parameters
    ----------
    yps: yprofile objects
        yprofiles to compare
    labels : string array matching yp array
        labels for yp objects
    fname
        specific dump to access information from
        
    Example
    -------
    import ppm
    yp1 = yprofile('/data/ppm_rpod2/YProfiles/agb-entrainment-convergence/H1')
    yp2 = yprofile('/data/ppm_rpod2/YProfiles/O-shell-M25/D2')
    compare_entrained_material([yp1,yp2],['O shell','AGB'], fname = 271)
    '''

    pl.close(ifig); fig = pl.figure(ifig)
    cb = utils.colourblind
    ls = utils.linestylecb
    me = 0.1
    yy = 0
    for yp in yps:
        pl.plot( yp.get('j',resolution='L', fname = fname),
              yp.get('FV H+He',resolution='L', fname = fname), \
              color = cb(yy), label=labels[yy],
              marker=ls(yy)[1], markevery = me,
              markeredgecolor = 'w')
        yy += 1
    pl.legend(loc=(.6,.1))

    pl.ylabel('$\mathrm{FV}\,\mathcal{F}_1$')
    pl.xlabel('grid cell number')


    ax=pl.gca()
    ax.set_yscale('log')
    #yticks = ax.yaxis.get_major_ticks()
    pl.yticks(10.**np.linspace(-12,0,7))
    fig.subplots_adjust( left = 0.17 )
    
########################################################################
# Plotting funcitons that dont have supported dependencies
########################################################################

def get_power_spectrum_RProfile(yprof_path, rprof_path, r0, t_lim=None, t_res=None, l_max=6):
    '''
    Calculates power spectra of radial velocity
    
    Parameters
    ----------
    yprof_path/rprof_path: strings
        paths for matching rprofile and yprofile files
    r0: float
        radius to examine
    t_lim: 2 index array
        [t_min, t_max], data to look at
    t_res: int
        set number of data points between [t_min, t_max], None = ndumps
    l_max:
        maximum spherical harmonic degree l
        
    Output
    ------
    t,l,power : time, spherical harmonic degree l, and power vectors
    '''
    yprof = ppm.yprofile(yprof_path, filename_offset=-1)
    n_buckets = 80
    rp_set = rprofile.rprofile_set(rprof_path)
    n_dumps = rp_set.dumps[-1]
        
    if t_lim is None:
        t_min = rp_set.get_dump(1).get('time')
        t_max = rp_set.get_dump(n_dumps).get('time')
        t_lim = [t_min, t_max]

    if t_res is None:
        t_res = n_dumps
        
    t = np.linspace(t_lim[0], t_lim[1], t_res)
    
    l = np.arange(0, l_max + 1)
    power = np.zeros((len(t), l_max + 1))
    
    for i in range(len(t)):
        dump = int(yprof.get('Ndump', fname=t[i], numtype='t', silent=True)[-1])
        if dump > n_dumps:
            dump = n_dumps
        rp = rp_set.get_dump(dump)
        r = rp.get_table('y')
        idx0 = np.argmin(np.abs(r - r0))
        
        vx = rp.get_table('ux')
        vy = rp.get_table('uy')
        vz = rp.get_table('uz')
        centers = rp.get_centers()
    
        lat = np.zeros(n_buckets)
        lon = np.zeros(n_buckets)
        vr = np.zeros(n_buckets)
        for bucket in range(n_buckets):
            x = centers[0, bucket]
            y = centers[1, bucket]
            z = centers[2, bucket]
            bucket_r = (x**2 + y**2 + z**2)**0.5
            lat[bucket] = 90. - (180./np.pi)*np.arccos(z/bucket_r)
            lon[bucket] = (180./np.pi)*np.arctan2(y, x)

            r_norm = np.array([x, y, z])/bucket_r
            v = np.array([vx[0, idx0, bucket+1], \
                          vy[0, idx0, bucket+1], \
                          vz[0, idx0, bucket+1]])
            vr[bucket] = np.dot(v, r_norm)
        
        coeffs, _ = SHExpandLSQ(vr, lat, lon, l_max)
        power[i, :] = spectrum(coeffs, convention='power', unit='per_l')

    return t, l, power

def plot_power_spectrum_RProfile(t, l, power, ifig=1, title='', vmin=1e-2, vmax=1.):
    '''
    Plots power spectra of radial velocity 
    
    Parameters
    ----------
    t,l,power : arrays
        time, sperical harmonic degree and power generated by
        get_power_spectrum_RProfile()
    title : string
        title
    '''
    ifig = ifig; pl.close(ifig); pl.figure(ifig, figsize=(8., 5.), dpi=125)
    extent = (t[0]/60., t[-1]/60., l[0] - 0.5, l[-1] + 0.5)
    aspect = 0.5*(extent[1] - extent[0])/(extent[3] - extent[2]) 
    max_power = np.max(power)
    norm = LogNorm(vmin=vmin*max_power, vmax=vmax*max_power, clip=True)
    #norm = Normalize(vmin=0., vmax=max_power, clip=True)
    pl.imshow(np.transpose(np.abs(power)), origin='lower', extent=extent, aspect=aspect, \
               interpolation='nearest', norm=norm, cmap='viridis')
    cb = pl.colorbar()
    cb.set_label('m$^2$ (s$^2$ l)$^{-1}$')
    pl.xlabel('t / min')
    pl.ylabel('l')
    pl.title(title, y=1.025)
    ax0= pl.gca()
    ax0.get_yaxis().set_tick_params(direction='out')
    ax0.get_xaxis().set_tick_params(direction='out')

def bucket_map(rprofile, quantity, limits = None, ticks = None, file_name = None, time = None):
    
    '''
    Plots a Mollweide projection of the rprofile object using the mpl_toolkits.basemap package
    
    Parameters:
    -----------
    rprofile: rprofile object
        rprofile dump used just for geometry
    quantity: array
        data to be passed into the projection
    limits: 2 index array
        cmap limits, scale the colormap for limit[0] =min to 
        limit[1] =max
    ticks: 
        passed into matplotlib.colors.ColorbarBase see ColourbarBase
    file_name: string
        file name: '/path/filename' to save the image as
    time: float
        time to display as the title
    
    '''
    q = quantity#rp.get_table(quantity)[0, :, :]
    #r = rp.get('y')
    #idx_r0 = np.argmin(np.abs(r - r0))
    
    corners = rprofile.get_corners()
    corners_per_bucket = corners.shape[1]
    n_buckets = corners.shape[2]
    points_per_side = 10
    points_per_bucket = corners_per_bucket*points_per_side

    x = np.zeros((n_buckets, points_per_bucket))
    y = np.zeros((n_buckets, points_per_bucket))
    z = np.zeros((n_buckets, points_per_bucket))
    t = np.linspace(1., 0., num = points_per_side)
    for i in range(n_buckets):
        for k in range(corners_per_bucket):
            idx_range = list(range(points_per_side*k, points_per_side*(k + 1)))
            x[i, idx_range] = t*corners[0, k - 1, i] + (1. - t)*corners[0, k, i]
            y[i, idx_range] = t*corners[1, k - 1, i] + (1. - t)*corners[1, k, i]
            z[i, idx_range] = t*corners[2, k - 1, i] + (1. - t)*corners[2, k, i]

    radius = (x**2 + y**2 + z**2)**0.5
    phi = np.arctan2(y, x)
    theta = np.pi/2. - np.arccos(z/radius)

    eps = 1e-3
    for i in range(phi.shape[0]):
        for k in range(phi.shape[1] - 1):
            # if the vertex k is at one of the poles
            if (np.abs(theta[i, k] - 0.5*np.pi) < eps or
                np.abs(theta[i, k] + 0.5*np.pi) < eps):
                if (theta[i, k] == theta[i, k - 1] and
                    phi[i, k] == phi[i, k - 1]):
                    phi[i, k - 1] = phi[i, k - 2]
                    phi[i, k] = phi[i, k + 1]

    # A first estimate of how many rows will be needed. We need more
    # than n_buckets, because we have to slice the polygons that
    # lie at the boundaries of the plot.
    n_rows_est = int(np.round(1.25*n_buckets))
    phi2 = np.zeros((n_rows_est, points_per_bucket))
    theta2 = np.zeros((n_rows_est, points_per_bucket))
    value = np.zeros(n_rows_est)
    n_rows = 0
    for i in range(n_buckets):
        # Add more rows if necessary.
        if n_rows >= phi2.shape[0]:
            n_rows_add = int(np.round(0.25*phi2.shape[0]))
            phi2 = np.vstack((phi2, np.zeros((n_rows_add, points_per_bucket))))
            theta2 = np.vstack((theta2, np.zeros((n_rows_add, points_per_bucket))))
            value = np.append(value, np.zeros(n_rows_add))

        this_phi = np.copy(phi[i, :])
        this_theta = np.copy(theta[i, :])
        this_value = q[i]# np.log10(q[idx_r0, i])

        if not (np.min(this_phi) < -0.5*np.pi and np.max(this_phi) > 0.5*np.pi):
            # This polygon doesn't touch the boundaries of the plot. Original
            # coordinates can be used directly.
            phi2[n_rows, :] = this_phi
            theta2[n_rows, :] = this_theta
            value[n_rows] = this_value
            n_rows += 1
        else:
            # This polygon lies on the boundary of the plot. We have to slice into
            # two polygons -- one on the left side of the plot and on on the right.
            # First add the one on the right.
            this_phi2 = np.copy(this_phi)
            for k in range(points_per_bucket):
                if this_phi2[k] <= -0.:
                    this_phi2[k] = np.pi

            phi2[n_rows, :] = this_phi2
            theta2[n_rows, :] = this_theta
            value[n_rows] = this_value
            n_rows += 1

            # Now add the one on the left.
            this_phi2 = np.copy(this_phi)
            for k in range(points_per_bucket):
                if this_phi2[k] >= 0.:
                    this_phi2[k] = -np.pi

            phi2[n_rows, :] = this_phi2
            theta2[n_rows, :] = this_theta
            value[n_rows] = this_value
            n_rows += 1

    # Trim the arrays to the actual size of the data.
    if n_rows < phi2.shape[0]:
        phi2 = phi2[0:n_rows, :]
        theta2 = theta2[0:n_rows, :]
        value = value[0:n_rows]

    #ifig = 1; plt.close(ifig); fig = plt.figure(ifig, figsize = (9, 4))
    #ifig = 1; plt.close(ifig); fig = plt.figure(ifig, figsize = (3.39, 2.4))
    pl.clf()
    gs = gridspec.GridSpec(2, 1, height_ratios = [12, 1]) 
    ax0 = pl.subplot(gs[0])
    ax1 = pl.subplot(gs[1])
    m = Basemap(projection = 'moll', lon_0 = 0., ax = ax0)
    
    cmap_min = np.min(quantity)
    cmap_max = np.max(quantity)
    cmap_avg = np.sum(quantity)/float(len(quantity))
    
    if limits is not None:
        cmap_min = limits[0]
        cmap_max = limits[1]
        
        cmap_avg = 0.5*(cmap_min + cmap_max)
        
        if len(limits) > 2:
            cmap_avg = limits[2]
            
    cmap_avg_rel = (cmap_avg - cmap_min)/(cmap_max - cmap_min)
    gamma = 1.0
    c1 = np.array([95, 158, 209])/255.
    c2 = np.array([255, 255, 255])/255.
    c3 = np.array([200, 82, 0])/255.
    cmap_points = {'red':   ((0.0, 0.0, c1[0]),
                            (cmap_avg_rel**gamma, c2[0], c2[0]),
                            (1.0, c3[0], 0.0)),

                   'green': ((0.0, 0.0, c1[1]),
                            (cmap_avg_rel**gamma, c2[1], c2[1]),
                            (1.0, c3[1], 0.0)),

                   'blue':  ((0.0, 0.0, c1[2]),
                            (cmap_avg_rel**gamma, c2[2], c2[2]),
                            (1.0, c3[2], 0.0))
                  }
    cmap = LinearSegmentedColormap('my_cmap', cmap_points, gamma = gamma)
    #cmap_name = 'gist_earth_r'
    #cmap = plt.get_cmap(cmap_name)
    for i in range(phi2.shape[0]):
        t = (value[i] - cmap_min)/(cmap_max - cmap_min)
        if t < 0: t = 0.
        if t > 1: t = 1.
        facecolor = cmap(t)
        x, y = m((180./np.pi)*phi2[i, :], (180./np.pi)*theta2[i, :])
        xy = list(zip(x, y))
        poly = Polygon(xy, facecolor = facecolor, edgecolor = facecolor, lw = 0.25)
        ax0.add_patch(poly)
    #m.drawmapboundary(color = 'k', linewidth = 1.5)
    m.drawmapboundary(color = 'k', fill_color = 'none', zorder = 10000)
    #ax0.set_title(cmap_name)
    if time is not None:
        ax0.set_title('t = {:.0f} min'.format(time/60.))
    
    def fmt(x, pos):
        return '{: .2f}'.format(x)
    
    norm = matplotlib.colors.Normalize(vmin = cmap_min, vmax = cmap_max)
    cb = ColorbarBase(ax1, cmap = cmap, norm = norm, ticks = ticks, \
                      format = ticker.FuncFormatter(fmt), orientation='horizontal')
    cb.set_label(r'$\Delta$r$_\mathrm{ub}$ / Mm')
    #ropplt.tight_layout(h_pad = 2.)
    pl.show()
    if file_name is not None:
        #plt.savefig(file_name + '_' + cmap_name + '.pdf', bbox_inches = 'tight', facecolor = 'w', dpi = 332.7)
        pl.savefig(file_name, bbox_inches = 'tight', facecolor = 'w', dpi = 332.7)

def plot_Mollweide(rp_set, dump_min, dump_max, r1, r2, output_dir = None, Filename = None, ifig = 2):
    
    '''
    Plot Mollweide spherical projection plot
    
    Parameters
    ----------
    dump_min/dump_max = int
        Range of file numbers you want to use in the histogram
    r1/r2 = float
        This function will only search for the convective 
        boundary in the range between r1/r2
    ouput_dir: string
        path to output directory
    filename: string
        name for output file, None: no output
    
    Example
    -------
    data_path = "/rpod2/PPM/RProfiles/AGBTP_M2.0Z1.e-5/F4"
    rp_set = rprofile.rprofile_set(data_path)
    plot_Mollweide(rp_set, 100,209,7.4,8.4)
    '''
    
    pl.close(ifig); fig = pl.figure(ifig, figsize = (3.384, 2.))
    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(1.)
    dr_ub_avg = np.zeros(80)

    n = 0
    for dump in range(dump_min, dump_max + 1):
        rp = rp_set.get_dump(dump)
        res = analyse_dump(rp, r1, r2)
        r_ub = res[3]
        dr_ub = r_ub - sum(r_ub)/float(len(r_ub))
        dr_ub_avg += dr_ub
        n += 1

    dr_ub_avg /= float(n)
    if Filename is not None:
        filename = output_dir + Filename
    else:
        filename = None
        
    bucket_map(rp, dr_ub_avg, file_name = filename)

def get_mach_number(rp_set,yp,dumps):
    '''
    Returns max Mach number and matching time vector
    
    Parameters
    ----------
    yp: yprofile instance
    rp_set: rprofile set instance
    dumps: range
        range of dumps to include
        
    Returns
    -------
    Ma_max : vector
        max mach nunmber
    t : vector
        time vector
    '''
    nd = len(dumps)
    t = np.zeros(nd)
    Ma_max = np.zeros(nd)

    for i in range(nd):
        rp = rp_set.get_dump(dumps[i])
        t[i] = yp.get('t', fname = dumps[i] - 1, resolution = 'l')[-1]
        Ma_max[i] = np.max(rp.get_table('mach')[2, :, 0])
    
    return Ma_max, t

def plot_mach_number(rp_set,yp,dumps,ifig = 1,lims = None,insert = False,save=False,
                      prefix='PPM',format='pdf',lims_insert =None):
    '''
    Parameters
    ----------
    yp: yprofile instance
    rp_set: rprofile set instance
    dumps: range
        range of dumps to include
    lims : list
        Limits for the plot, i.e. [xl,xu,yl,yu].
        If None, the default values are used.
        The default is None.
    save : boolean
        Do you want the figures to be saved for each cycle?
        Figure names will be <prefix>-Vel-00000000001.<format>,
        where <prefix> and <format> are input options that default
        to 'PPM' and 'pdf'.
        The default value is False.
    prefix : string
        see 'save' above
    format : string
        see 'save' above
    '''
    try:
        Ma_max,t = get_mach_number(rp_set,yp,dumps)
    except:
        print('Dumps range must start at a value of 1 or greater')
    ifig = ifig; pl.close(ifig); fig = pl.figure(ifig)
    ax1 = fig.add_subplot(111)
    ax1.plot(t/60., Ma_max, color=cb(3))
    if lims is not None:
        ax1.set_xlim((lims[0],lims[1]))
        ax1.set_ylim((lims[2],lims[3]))
    ax1.set_xlabel('t / min')
    ax1.set_ylabel(r'Ma$_\mathrm{max}$')
    
    if insert:
        left, bottom, width, height = [0.27, 0.55, 0.3, 0.3]
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.plot(t/60., Ma_max, color=cb(3))
        if lims_insert is not None:
            ax2.set_xlim((lims_insert[0],lims_insert[1]))
            ax2.set_ylim((lims_insert[2],lims_insert[3]))
    fig.tight_layout()

    if save:
        pl.savefig(prefix+'_Ma_max_evolution.'+format,format=format)

def get_p_top(prof,dumps,r_top):
    '''
    Returns p_top and matching time vector
    
    Parameters
    ----------
    yp: yprofile instance
    rtop: float
        boundary radius
    dumps: range
        range of dumps to include can be sparse
        
    Returns
    -------
    Ma_max : vector
        max mach nunmber
    t : vector
        time vector
    '''
    nd = len(dumps)
    t = np.zeros(nd)
    p_top = np.zeros(nd)

    r = prof.get('Y', fname = 0, resolution = 'l')
    idx_top = np.argmin(np.abs(r - r_top))

    for i in range(nd):
        t[i] = prof.get('t', fname = dumps[i], resolution = 'l')[-1]
        p_top[i] = prof.get('P', fname = dumps[i], resolution = 'l')[idx_top]

    return p_top,t

def plot_p_top(yp,dumps,r_top,ifig = 1,lims = None,insert = False,save=False,
                      prefix='PPM',format='pdf',lims_insert =None):
    '''
    Parameters
    ----------
    yp: yprofile instance
    r_top: float
        boundary radius
    dumps: range
        range of dumps to include can be sparse
    lims : list
        Limits for the plot, i.e. [xl,xu,yl,yu].
        If None, the default values are used.
        The default is None.
    save : boolean
        Do you want the figures to be saved for each cycle?
        Figure names will be <prefix>-Vel-00000000001.<format>,
        where <prefix> and <format> are input options that default
        to 'PPM' and 'pdf'.
        The default value is False.
    prefix : string
        see 'save' above
    format : string
        see 'save' above
    '''

    p_top,t = get_p_top(yp,dumps,r_top)
    ifig = ifig; pl.close(ifig); fig = pl.figure(ifig)
    ax1 = fig.add_subplot(111)
    ax1.plot(t/60., p_top/p_top[0] - 1., color=cb(3))
    if lims is not None:
        ax1.set_xlim((lims[0],lims[1]))
        ax1.set_ylim((lims[2],lims[3]))
    ax1.set_xlabel('t / min')
    ax1.set_ylabel(r'p$_\mathrm{top}$(t) / p$_\mathrm{top}$(0) - 1')
    
    if insert:
        left, bottom, width, height = [0.31, 0.55, 0.3, 0.3]
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.plot(t/60., p_top/p_top[0] - 1., color=cb(3))
        if lims_insert is not None:
            ax2.set_xlim((lims_insert[0],lims_insert[1]))
            ax2.set_ylim((lims_insert[2],lims_insert[3]))
    fig.tight_layout()

    if save:
        pl.savefig(prefix+'_p_top_evolution.'+format,format=format)    
#####################################################
# Files with mesa dependancy
#####################################################

def get_N2(yp, dump):
    '''
    squared Brunt-Vaisala frequency N^2

    Parameters
    ----------
    yp = yprofile object
        yprofile to use
    dump = int
        dump to analyze
    
    Output
    ------
    N2 = vector
        Brunt Vaisala frequency [rad^2 s^-1]
    '''
    
    G_code = ast.grav_const/1e-3
    nabla_ad = 0.4
    
    R_bot = float(yp.hattrs['At base of the convection zone R'])
    g_bot = float(yp.hattrs['At base of the convection zone g'])
    
    r = yp.get('Y', fname = dump, resolution = 'l')
    p = yp.get('P', fname = dump, resolution = 'l')
    rho = yp.get('Rho', fname = dump, resolution = 'l')
    
    # centre R_bot on the nearest cell
    idx_bot = np.argmin(np.abs(r - R_bot))
    
    # mass below R_bot
    # (using r[idx_bot] instead of R_bot makes g[idx_bot] == g_bot)
    M_bot = g_bot*(r[idx_bot]**2)/G_code
    dr = 0.5*(np.roll(r, -1) - np.roll(r, +1))
    dm = 4.*np.pi*(r**2)*dr*rho
    m = np.cumsum(dm) # get the mass profile by integration
    # shift the mass profile to make sure that m[idx_bot] == M_bot
    # the mass profile at small radii won't make sense, because
    # the core is artificial with no gravity
    m += M_bot - m[idx_bot]
    
    g = G_code*m/(r**2) # gravity profile (see the note above)
    H_p = p/(rho*g) # pressure scale height (assuming hydrostatic equilibrium)
    
    logrho = np.log(rho)
    dlogrho = 0.5*(np.roll(logrho, -1) - np.roll(logrho, +1))
    logp = np.log(p)
    dlogp = 0.5*(np.roll(logp, -1) - np.roll(logp, +1))
    nabla_rho = dlogrho/dlogp
    
    # N^2 for an ideal gas
    N2 = (g/H_p)*(nabla_ad - 1. + nabla_rho)
    
    return N2

def plot_N2(case, dump1, dump2, lims1, lims2, mesa_A_model_num):

    '''
        plots squared Brunt-Vaisala frequency N^2 with zoom window

        Parameters
        ----------
        case = string 
            Name of run eg. 'D1'
        dump1/ dump2 = int
            dump to analyze
        mesa_A_model_num = int
            number for mesa model
        lims1/lims2 = 4 index array
            axes limits [xmin xmax ymin ymax] lims1 = smaller window

        Example
        ------
        import ppm
        set_YProf_path('/data/ppm_rpod2/YProfiles/O-shell-M25/')
        plot_N2('D1', 0, 132, mesa_A_model_num = 29350, mesa_B_model_num = 28950)
        '''
    ppm_run= case
    yp = yprofile(ppm_path + case)
    mesa_logs_path = '/data/ppm_rpod2/Stellar_models/O-shell-M25/M25Z0.02/LOGS_N2b'
    mesa_A_prof = ms.mesa_profile(mesa_logs_path, mesa_A_model_num)
    # convert the mesa variables to code units
    mesa_A_r = (ast.rsun_cm/1e8)*mesa_A_prof.get('radius')
    mesa_A_N2 = mesa_A_prof.get('brunt_N2')
    mesa_A_N2_mu = mesa_A_prof.get('brunt_N2_composition_term')
    mesa_A_N2_T = mesa_A_prof.get('brunt_N2_structure_term')
    mesa_A_mu = mesa_A_prof.get('mu')

    # get the PPM models
    ppm_prof = yp
    ppm_r = ppm_prof.get('Y', fname = 0, resolution = 'l')
    ppm_dump_num_A = dump1
    ppm_t_A = ppm_prof.get('t', fname = ppm_dump_num_A, resolution = 'l')[-1]
    ppm_N2_A = get_N2(ppm_prof, ppm_dump_num_A)
    ppm_dump_num_B = dump2
    ppm_t_B = ppm_prof.get('t', fname = ppm_dump_num_B, resolution = 'l')[-1]
    ppm_N2_B = get_N2(ppm_prof, ppm_dump_num_B)

    cb = utils.colourblind

    ifig = 1; pl.close(ifig); fig = pl.figure(ifig)

    ax1 = fig.add_subplot(111)
    ax1.plot(mesa_A_r, mesa_A_N2, ls = '-', lw = 0.5, color = cb(4), label = "MESA")
    lbl = "{:s}, t = {:.0f} min.".format(ppm_run, ppm_t_A/60.)
    ax1.plot(ppm_r, ppm_N2_A, ls = ':', color = cb(5), label = lbl)
    lbl = "{:s}, t = {:.0f} min.".format(ppm_run, ppm_t_B/60.)
    ax1.plot(ppm_r, ppm_N2_B, ls = '-', color = cb(8), label = lbl)
    ax1.set_xlim((lims1[0], lims1[1]))
    ax1.set_ylim((lims1[2],lims1[3]))
    ax1.set_xlabel('r / Mm')
    ax1.set_ylabel(r'N$^2$ / rad$^2\,$s$^{-2}$')
    ax1.legend(loc = 0)

    left, bottom, width, height = [0.34, 0.3, 0.4, 0.4]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(mesa_A_r, mesa_A_N2, ls = '-', lw = 0.5, color = cb(4))
    ax2.plot(ppm_r, ppm_N2_A, ls = ':', color = cb(5))
    ax2.plot(ppm_r, ppm_N2_B, ls = '-', color = cb(8))
    ax2.set_xlim((lims2[0], lims2[1]))
    ax2.set_ylim((lims2[2],lims2[3]))

def energy_comparison(yprof,mesa_model, xthing = 'm',ifig = 2, silent = True,
                     range_conv1 = None , range_conv2 = None,
                     xlim = [0.5,2.5] , ylim = [8,13], radbase = 4.1297, 
                     dlayerbot = 0.5, totallum = 20.153):
    '''
    Nuclear  energy  generation  rate (enuc) thermal  neutrino   
    energy   loss   rate (enu) and   luminosity   profiles (L) for MESA
    plotted with ppmstar energy estimation (eppm)
    
    Parameters
    ----------
    yprof: yprofile object
        yprofile to examine
    mesa_model
        mesa model to examine
    xthing: string
        x axis as mass, 'm' or radius, 'r'
    silent: boolean
        suppress output or not
    range_conv1: range or None
        range to shade for convection zone 1
    range_conv2: range or None
        range to shade for convection zone 2
    xlim: range
    ylim: range
   
    values from setup.txt
    radbase = 4.1297
    dlayerbot = 0.5
    totallum = 20.153
    
    Example
    -------
    import ppm
    import nugrid.mesa as ms
    mesa_A_model_num = 29350
    mesa_logs_path = '/data/ppm_rpod2/Stellar_models/O-shell-M25/M25Z0.02/LOGS_N2b'
    mesa_model = ms.mesa_profile(mesa_logs_path, mesa_A_model_num)
    yprof = yprofile('/data/ppm_rpod2/YProfiles/O-shell-M25/D1')
    energy_comparison(yprof,mesa_model)
    
    '''
    xthing1 = xthing
    p = mesa_model
    # function giving PPM heating curve given PPM profile

    # figure vs mass
    ifig = 2; pl.close(ifig); fig = pl.figure(ifig)
    m = p.get('mass')

    eps_nuc = np.log10(p.get('eps_nuc'))
    eps_neu = np.log10(p.get('non_nuc_neu'))
    lum = p.get('logL')

    # plot conv zone boundaries
    if range_conv1 is not None:
        pl.fill_between( range_conv1 , 0, 100, color= '#fffdd8' )
        
    if range_conv2 is not None:
        pl.fill_between( range_conv2 , 0, 100, color='#b39eb5' )

    res = get_heat_source(yprof, radbase = radbase, dlayerbot = dlayerbot, totallum = totallum)
    r_ppm = res[:, 0]
    m_ppm = res[:, 1]
    m_ppm = m_ppm * 5.025e-07 # code unit --> solar masses
    eps_ppm = res[:, 2]
    eps_ppm = eps_ppm * 1.e16 # code units --> erg/g/s
    eps_ppm = np.log10(eps_ppm)

    # trim mess off eps_nuc below O-shell
    idx = np.abs(m - 1.0).argmin()

    # plot simulation boundaries
    rad1 = 3.0 # Mm
    rad2 = 9.8 # Mm
    rad  = 10. ** p.get('logR') * ast.rsun_cm / 1.e8
    if not silent:
        print(rad)

    idx1 = np.abs(rad - rad1).argmin()
    idx2 = np.abs(rad - rad2).argmin()
    m1   = m[idx1]
    m2   = m[idx2]

    if not silent:
        print(m1, m2)

    if xthing1 is 'm':
        xthing = m
        xthing_ppm = m_ppm
    else:
        xthing = r
        xthing_ppm = r_ppm

    #semilogy(m[:idx],eps_nuc[:idx],
    pl.plot(xthing[:idx],eps_nuc[:idx],
        label='$\epsilon_\mathrm{nuc}$',
        color=utils.colourblind(8),
        linestyle='-')

    #semilogy(m,eps_neu,
    pl.plot(xthing,eps_neu,
        label='$\epsilon_\\nu$',
        color=utils.colourblind(4),
        linestyle='--')

    #semilogy(m,lum,
    pl.plot(xthing,lum,
        label='$L$',
        color=utils.colourblind(0),
        linestyle='-.')

    #semilogy(m_ppm,eps_ppm,
    pl.plot(xthing_ppm,eps_ppm,
        label='$\epsilon_\mathrm{PPM}$',
        color=utils.colourblind(4),
        linestyle='-',
        marker='.',
        markevery=.3)

    #pl.ylabel('$\epsilon\,/\,\mathrm{erg\,g}^{-1}\,\mathrm{s}^{-1}\,;\,L\,/\,L_\odot$')
    if xthing1 is 'm':
        pl.ylabel('$\log_{10}(\epsilon\,/\,\mathrm{erg\,g}^{-1}\,\mathrm{s}^{-1}\,;\,L\,/\,L_\odot)$')
        pl.xlabel('$\mathrm{Mass\,/\,M}_\odot$')
    else:
        pl.ylabel('$\log_{10}(\epsilon\,/\,\mathrm{erg\,g}^{-1}\,\mathrm{s}^{-1}\,;\,L\,/\,L_\odot)$')
        pl.xlabel('$\mathrm{r\,/\,Mm}$')

    pl.legend(loc='upper left')

    #pl.xlim(1.,2.5)
    pl.xlim(xlim)
    #pl.ylim(1.e8,1.e13)
    pl.ylim(ylim)
    
def get_heat_source(yprof, radbase = 4.1297, dlayerbot = 0.5, totallum = 20.153):
    '''
    returns estimation of yprofiles energy
    
    # values from setup.txt
    radbase = 4.1297
    dlayerbot = 0.5
    totallum = 20.153
    
    Parameters
    ----------
    yprof: yprofile object
        yprofile to examine
        
    Returns
    -------
    array
        array with vectors [radius, mass, energy estimate]
    '''
    G_code = ast.grav_const/1e-3

    r = yprof.get('Y', fname = 0, resolution = 'l')
    rho = yprof.get('Rho', fname = 0, resolution = 'l')
    r_bot = float(yprof.hattrs['At base of the convection zone R'])
    g_bot = float(yprof.hattrs['At base of the convection zone g'])

    dr = -0.5*(np.roll(r, -1) - np.roll(r, +1))
    dr[0] = dr[1]; dr[-1] = dr[-2]
    
    # centre r_bot on the nearest cell
    idx_bot = np.argmin(np.abs(r - r_bot))
    # mass below r_bot
    m_bot = g_bot*(r[idx_bot]**2)/G_code

    dV = 4.*np.pi*(r**2)*dr
    dm = rho*dV
    m = -np.cumsum(dm) # get the mass profile by integration
    # shift the mass profile to make sure that m[idx_bot] == m_bot
    m += m_bot - m[idx_bot]

    radminbase = radbase + 0.5*dlayerbot
    radmaxbase = radbase + 1.5*dlayerbot

    # PPMstar computes this integral numerically
    heatsum = 4.*pi*(-(radmaxbase**5 - radminbase**5)/5. + \
                     (radminbase + radmaxbase)*(radmaxbase**4 - radminbase**4)/4. - \
                     radminbase*radmaxbase*(radmaxbase**3 - radminbase**3)/3.)
    dist = np.maximum(0.*r, r - radminbase)*np.maximum(0.*r, radmaxbase - r)*dV/heatsum
    
    # the tiny offset of 1e-100 makes sure lines don't end suddenly in a logarithmic plot
    # 2.25 is the correction factor to account for the heating bug in PPMstar
    eps = 2.25*totallum*dist/dm + 1e-100
    
    return np.transpose(np.array([r, m, eps]))

def get_mesa_time_evo(mesa_path,mesa_logs,t_end,save = False):
    '''Function to generate data for figure 5 in O-shell paper.
    
    Parameters
    ----------
    mesa_path : string 
        path to the mesa data
    mesa_logs : range 
        cycles you would like to include in the plot
    t_end : float
        time of core collapse
    save : boolean,optional
        save the output into data files
        
    Returns (all arrays)
    -------
    agearr : age yrs
    ltlarr  : time until core collapse
    rbotarr : radius of lower convective boundary
    rtoparr : radius of top convective boundary
    muarr : mean molecular weight in convective region
    peakLarr : peak luminosity
    peakL_Lsunarr : peak luminosity units Lsun
    peakepsgravarr :
    '''
    tag = 'shell1'
    #tag = 'shell2'

    s = ms.history_data(mesa_path)
 
    #logs=range(825,1000)

    agearr         = []
    ltlarr         = []
    rbotarr        = []
    rtoparr        = []
    muarr          = []
    peakLarr       = []
    peakL_Lsunarr  = []
    peakepsgravarr = []

    # latex table file:
    if save:
        f=open('table.tex','w')
    for log in mesa_logs:
        p=ms.mesa_profile(mesa_path,log,num_type='profile_num')
        mass = p.get('mass')
        idxl = np.abs( mass - 1. ).argmin()
        idxu = np.abs( mass - 2. ).argmin()
        mass = mass[idxu:idxl]
        rad = 10.**p.get('logR')[idxu:idxl]
        mt = p.get('mixing_type')[idxu:idxl]
        L = 10.**p.get('logL')[idxu:idxl]
        epsgrav = p.get('eps_grav')[idxu:idxl]
        peakL_Lsun = np.max(L)/1.e10
        peakL = peakL_Lsun*1.e10*ast.lsun_erg_s/1.e44
        peakepsgrav = np.max(epsgrav)
        ipL = L.argmax()
        mu = p.get('mu')[idxu:idxl]
        try:
            itop = np.where(mt[:ipL]!=1)[0][-1]
        except:
            continue
        rtop = rad[:ipL][itop]*ast.rsun_cm/1.e8
        mtop = mass[itop]
        ibot = np.where(mt==1)[0][-1]
        rbot = rad[ibot]*ast.rsun_cm/1.e8
        mbot = mass[ibot]
        mu = mu[(itop+ibot)/2]
        # time from end of core O burning
        iaoe = np.where(s.get('center_o16')>1e-3)[0][-1] #was 's'?
        aoe = s.get('star_age')[iaoe] #was 's'?
        age = ( p.header_attr['star_age'] - aoe ) * 365.
        ltl = np.log10( t_end - p.header_attr['star_age'] )

        agearr.append( age )
        ltlarr.append( ltl )
        rbotarr.append( rbot )
        rtoparr.append( rtop )
        muarr.append( mu )
        peakLarr.append( peakL )
        peakepsgravarr.append( peakepsgrav )
        peakL_Lsunarr.append( peakL_Lsun )
        if save:
            for x in [age,ltl,rbot,rtop,mu,peakL,peakL_Lsun]:
                f.write("{0:.3f}".format(x) + ' & ')
            f.write(' \\ \n')
    if save:
        f.close()

        #save arrays for making some plots
        np.save( tag+'_age.npy', np.array( agearr ) )
        np.save( tag+'_ltl.npy', np.array( ltlarr ) )
        np.save( tag+'_rbot.npy', np.array( rbotarr ) )
        np.save( tag+'_rtop.npy', np.array( rtoparr ) )
        np.save( tag+'_mu.npy', np.array( muarr ) )
        np.save( tag+'_peakL.npy', np.array( peakLarr ) )
        np.save( tag+'_peakL_Lsun.npy', np.array( peakL_Lsunarr ) )
        np.save( tag+'_peakepsgrav.npy', np.array( peakepsgravarr ) )

    return agearr,ltlarr,rbotarr,rtoparr,muarr,peakLarr,peakL_Lsunarr,peakepsgravarr

def plot_mesa_time_evo(mesa_path,mesa_logs,t_end,ifig=21):
    '''
    Function to plot data for figure 5 in O-shell paper.
    
    Parameters
    ----------
    mesa_path : string 
        path to the mesa data
    mesa_logs : range 
        cycles you would like to include in the plot
    t_end : float
        time of core collapse
    save : boolean,optional
        save the output into data files
        
    Example
    -------
    plot_mesa_time_evo('/data/ppm_rpod2/Stellar_models/O-shell-M25/M25Z0.02/LOGS',
                   range(550,560),7.5829245098141646E+006,ifig=21)
    '''
    cb = utils.colourblind
    #pl.close(ifig),pl.figure(ifig)
    f, (ax1, ax2, ax3) = pl.subplots(3, sharex=True, figsize=(3.39,5))
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0.1)
    pl.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    # load arrays
    
    data = get_mesa_time_evo(mesa_path,mesa_logs,t_end)
    
    age = data[0]
    ltl = data[1]
    mu = data[4]
    peakL_Lsun = data[6]
    peakL = data[5]
    rbot = data[2]
    rtop = data[3]
    epsgrav = data[7]
    '''
    age = np.load(tag+'age.npy')
    ltl = np.load(tag+'ltl.npy')
    mu = np.load(tag+'mu.npy')
    peakL_Lsun = np.load(tag+'peakL_Lsun.npy')
    peakL = np.load(tag+'peakL.npy')
    rbot = np.load(tag+'rbot.npy')
    rtop = np.load(tag+'rtop.npy')
    epsgrav = np.load(tag+'peakepsgrav.npy')
    '''
    xax = ltl
    #xax = age

    #xlim(-1.4,-1.9)

    # make plots
    pl.subplot(ax1)
    pl.plot(xax,mu,color=cb(5))
    pl.ylabel('$\mu$')
    #pl.ylim(1.82,1.88)
    #pl.yticks(linspace(1.82,1.88,4))

    pl.subplot(ax2)
    pl.plot(xax, rbot, color=cb(5), label='$r_\mathrm{lb}$')
    pl.plot(xax, rtop, ls='--', color=cb(8), label='$r_\mathrm{ub}$')
    #pl.ylim(3,9)
    #pl.yticks(np.linspace(3,9,4))
    pl.ylabel('$\mathrm{r\,/\,Mm}$')
    pl.legend(loc='best')

    pl.subplot(ax3)
    pl.plot(xax, peakL_Lsun, color=cb(5) )
    pl.ylabel('$L\,/\,10^{10}L_\odot$')
    pl.xlabel('$\log_{10}(t^*\,/\,\mathrm{yr})$')
    #pl.ylim(4,8)
    #pl.yticks(linspace(4,8,5))

    pl.subplots_adjust(bottom=0.1)

def plot_entr_v_ut(cases,c0, Ncycles,r1,r2, comp,metric,label, ifig0 = 3,
                  integrate_both_fluids = False):
    '''
    Plots entrainment rate vs max radial or tangential velocity
    
    Parameters
    ----------
    cases : string array
        list of cases i.e.D1 
    c0 : int array
        cycle to start on for each case
    Ncycles: int
        number of cycles to average to find v
    r1/r2 :floats
        radius range to search for v
    comp : string
        component of velocity 'tangential' or 'radial'
    metric : string
        metric of veloctiy 'min' max' or 'mean'
        
    Example
    -------
    cases = ('D1', 'D8', 'D5', 'D6', 'D9', 'D10', 'D20', 'D23', 'D2')
    c0 = (241,154,142,155,123,78,355,241,124)
    global ppm_path
    ppm_path = '/data/ppm_rpod2/YProfiles/O-shell-M25/'    
    plot_entr_v_ut(cases,c0,10,7.5,8.5,ifig0 = 3,
                      integrate_both_fluids = False)
    '''
    mdot = np.zeros(len(cases))
    vt = np.zeros(len(cases))
    vr = np.zeros(len(cases))

    for i in range(len(cases)):
        prof = yprofile(ppm_path + cases[i])
        cycles = list(range(c0[i], c0[i] + Ncycles, 1))
        #vt[i], vr[i] = find_max_velocities(prof, cycles, 7.5, 8.5, 4., 8., label = cases[i], ifig = i)
        t, v = get_v_evolution(prof, cycles, r1,r2,comp = comp, RMS = metric)
        vt[i] = np.mean(1e3*v)
        mdot[i] = prof.entrainment_rate(cycles, r1, r2, var = 'vxz',
                                        criterion = 'min_grad', offset = -1.,
                                        integrate_both_fluids = integrate_both_fluids,
                                        ifig0 = ifig0,
                                        show_output = False)
        
    fc = np.polyfit(np.log(vt[0:len(cases)]/30.), np.log(mdot[0:len(cases)]), 1)
    vt_fit = np.array((1e0, 1e3))
    mdot_fit = np.exp(fc[0]*np.log(vt_fit/30.) + fc[1])

    mdot0_str = '{:9e}'.format(np.exp(fc[1]))
    tmp = mdot0_str.split('e')
    mantissa = float(tmp[0])
    exponent = int(tmp[1])
    fit_label = r'${:.2f} \times 10^{{{:d}}}$ (v$_\perp$ / 30)$^{{{:.2f}}}$'.\
            format(mantissa, exponent, fc[0])
    nn = len(cases)
    cb = utils.colourblind
    ifig = 1; pl.close(ifig); pl.figure(ifig)
    pl.plot(np.log10(vt[0:nn]), np.log10(mdot[0:nn]),
             ls = 'none', color = cb(5), marker = 'o', \
             label = label)
    pl.plot(np.log10(vt_fit), np.log10(mdot_fit),
             ls = '-', lw = 0.5, color = cb(4), \
             label = fit_label)
    pl.plot(np.log10(59.), np.log10(1.1e-4), ls = 'None',
             color = cb(8), marker = 's', \
             label = 'MA07')
    pl.xlabel(r'log$_{10}$(v$_\perp$ / km s$^{-1}$)')
    pl.ylabel(r'log$_{10} ( \dot{\mathrm{M}}_\mathrm{e}$ / M$_\odot$ s$^{-1}$])')
    #plt.xlim((0.9, 2.3))
    #plt.ylim((-9., -3.))
    pl.legend(loc = 4)
    pl.tight_layout()

def plot_diffusion_profiles(run,mesa_path,mesa_log,rtop,Dsolve_range,tauconv,r0,D0,f1,f2,
                            alpha,fluid = 'FV H+He',markevery = 25):
    '''
    This is a function that generates various diffusion profiles
    it is dependant on the functions Dsolvedown and Dov2
    
    Parameters
    ----------
    run : string
        yprofile case to use i.e. 'D2'
    mesa_path : string 
        path to the mesa data
    mesa_logs : range 
        cycles you would like to include in the plot
    Dsolve_range : int array 
        cycles from which to take initial and final abundance profiles
        for the diffusion step we want to mimic.
    fluid : string
        Which fluid do you want to track?
    tauconv : float, optional
        If averaging the abundance profiles over a convective turnover
        timescale, give the convective turnover timescale (seconds).
        The default value is None.
    r0 : float
        radius (Mm) at which the decay will begin
    D0 : float
        diffusion coefficient at r0
    f1,f2 : float
        parameters of the model
        
    Example
    --------
    plot_diffusion_profiles('D2','/data/ppm_rpod2/Stellar_models/O-shell-M25/M25Z0.02/',28900,
                        7.8489,(1,160),2.*460.,7.8282,10.**12.27,0.21,0.055,1.6)
    pl.xlim(3,9)
    pl.ylim(11,16)
    '''
    
    run = 'D2'
    cb = utils.colourblind
    ls = utils.linestyle
    model = mesa_log
    dir  = mesa_path
    yy = 0
    YY = yprofile(ppm_path+run)

    [rrc, DDc, r0c, rc] = YY.Dsolvedown(Dsolve_range[0],Dsolve_range[1], fluid = fluid,
                                        tauconv = tauconv,returnY = True, newton = True,
                                        smooth = False,showfig = False)

    rrf2, DDf2 = YY.Dov2(r0,D0,f1,f2) # r0, D0, f1, f2

    r = YY.get('Y',fname=1,resolution='l')[::-1]
    vav = YY.vaverage('vY')[::-1] # cm / s

    P = YY.get('P',fname=1)[::-1] * 1.e19 # barye, centre to surface
    Hp = - P[1:] * np.diff(r) * 1.e8 / np.diff(P)
    Hp = np.insert(Hp,0,0)
    Dav = (1./3.) * vav * alpha * Hp

    p=ms.mesa_profile(dir+'/LOGS',model)
    rm = p.get('radius') * ast.rsun_cm / 1.e8
    idx = np.abs(rm-rtop).argmin()
    rm = rm[idx:]
    Dm = p.get('log_mlt_D_mix')
    Dm = Dm[idx:]

    v_mlt = 10.**p.get('log_conv_vel')[idx:]
    Hpmes = p.get('pressure_scale_height')[idx:]*ast.rsun_cm
    Davmes = (1./3.) * v_mlt * np.minimum(alpha * Hpmes,rtop*1.e8-rm*1.e8)

    Dav2 = (1./3.) * vav * np.minimum(alpha * Hp,rtop*1.e8-r*1.e8)

    fig = pl.figure(21)
    pl.plot(rrc,np.log10(DDc),color = cb(yy),markevery=markevery,\
            label='$D_{\\rm conv}$')
    yy +=1
    pl.plot(rm,Dm,color = cb(yy),marker = ls(yy)[1] ,markevery=markevery,\
         label='$D_{\\rm MLT}$')
    yy +=1

    pl.plot(r,np.log10(Dav),color = cb(yy),marker = ls(yy)[1], markevery=markevery,\
         label='$\\frac{1}{3}v\ell$')
    yy +=1

    pl.plot(r,np.log10(Dav2),color = cb(yy),marker = ls(yy)[1],markevery=markevery,linewidth=2.5,alpha=0.5,\
         label='$\\frac{1}{3}v{\\rm min}(\ell,r_0-r)$')
    yy +=1
    pl.plot(rm,np.log10(Davmes),color = cb(yy),marker = ls(yy)[1],markevery=markevery,linewidth=2.5,alpha=0.5,\
         label='$\\frac{1}{3}v_{\\rm MLT}{\\rm min}(\ell,r_0-r)$')
    pl.legend(loc='center left',numpoints=1).draw_frame(False)
    pl.ylabel('$\log(D\,/\,{\\rm cm}^2\,{\\rm s}^{-1})$')
    pl.xlabel('r / Mm')