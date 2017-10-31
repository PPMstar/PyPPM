#from ppmpy import ppm
data_dir = '/data/ppm_rpod2/YProfiles/'
project = 'O-shell-M25'
ppm.set_YProf_path(data_dir+project)

D2=ppm.yprofile('D2')
D2.prof_time([0,5,10],logy=False,num_type='time')