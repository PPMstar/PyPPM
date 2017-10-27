import ppmpy.ppm as ppm
D1 = ppm.yprofile('/data/ppm_rpod2/YProfiles/O-shell-M25/D1')
I4 = ppm.yprofile('/data/ppm_rpod2/YProfiles/C-ingestion/I4')
D1.vprofs( [240,330],fname_type = 'range', initial_conv_boundaries = False, ifig = 3, lw=3.,run='D1')
I4.vprofs( [270,372],fname_type = 'range', initial_conv_boundaries = False,
          ifig = 3, lw=1.,run='I4',lims=[3.5,9.2,0,50])