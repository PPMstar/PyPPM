from ppmpy import ppm
data_dir = '/data/ppm_rpod2/YProfiles/'
project = 'O-shell-M25'
ppm.set_YProf_path(data_dir+project)

D2=ppm.yprofile('D2')
D1=ppm.yprofile('D1')
ppm.prof_compare([D2,D1])