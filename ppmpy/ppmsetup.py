'''pm setup tools

This module provides a number of utilities that support the 
construction of a new setup for the PPMstar code. 

FH, 20140907

OC, 20180414 , 20190211
'''
from nugridpy import mesa as ms
import nugridpy.constants
from ppmpy import ppm
import numpy as np
import sys
import re
from nugridpy.ascii_table import readTable

G_code = ppm.G_code
a_cgs  = nugridpy.constants.radiation_const
a_code = a_cgs * 10**17
R_cgs  = nugridpy.constants.boltzmann_const*nugridpy.constants.avogadro_const
R_code = R_cgs /1.0e7


def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d'%(ix, iy))

    global coords
    coords.append((ix, iy))

    if len(coords) == npoints_select:
        fig.canvas.mpl_disconnect(cid)

    return coords


def reduce_h(r,r0=None):
    '''reduce double resolution array with error to single grid
    resolution'''

    if r0 is None:
        rreduc = (r[1::2]+r[0::2])/2.
    else:
        rreduc = r[::2]+r0[::2]

    return rreduc

def get_dm_and_dp(m,r1,r2,P,rho):
    '''
    calculate mass in shell between radius r1 and r2 (r2>r1)) with
    pressure P and assuming a polytopic rho-P relation with constant K
    and exponent gamma = 1/gammainv

    calculate dp in that shell
    '''

    rmid = 0.5 * (r1 + r2)
    dr   = r2 - r1
    dp   = -G_code * m * rho * dr / rmid**2
    dm   = rho * (4./3.) * pi * (r2**3 - r1**3) 
    return dm, dp, rmid

def Tppm(p_ppm,rho_ppm, airmu):
    '''
    calculate T in 10^9K from p and rho in code units from PPM star code

    for details of the code units see comments in PPM star code
    p_ppm is the total pressure
    rho_ppm is the density of the convective fluid, Rho_conv
    '''
    amuairbyR = airmu / R_code
    T9 = amuairbyR * p_ppm / rho_ppm
    return T9

def Pppm(T9,rho_ppm,mu):
    '''
    calculate P in code units from rho in code units from PPM star code
    and T in 10*9K

    for details of the code units see comments in PPM star code

        in:
    rho_ppm is the density of the convective fluid, Rho_conv
    T       temperature
    mu      mean   molecular weight

        out:
    p_ppm   total pressure
    '''
    
    mubyR = mu / R_code
    p_ppm = T9 *rho_ppm / mubyR 
    return p_ppm

def UnitConvert(datadir, quantity, convertto='PPMUnits', fromtype='mesaprof', modordump=1, filename=None):
    '''
    Converts from units used in MESA(cgs and solar) to PPMStar code units and vice versa.
    
    
    datadir, str:   path to mesa profile or rprofile to be read
    
    
    quantity, str : which quantity you want to convert. Options are density, radius, mass, 
                    pressure, temperature
    
    convertto, str: the unit system to convert to, default is PPMUnits but can choose MESAUnits.
    
    fromtype, str: what kind of data are you giving? Options are mesaprof, rprof or ppmsetup
    
    modndump, int : model number for mesa profile read if going fro mesa to ppm 
                    and ppm dump number if going from ppm to mesa.

    filename, str: default None, only needed if you are converting from type ppmsetup because ascii_table requires a dir and filename
    
    
    out: array in units converted to
    ''' 
    
    to_cgs = {'density': 10**3,
              'pressure': 10**19,
              'temperature': 10**9,
              'radius': 1.0 / 695.99}
    
    error_msg_quantity = "[%s] Quantity not recognized '%s'." % (convertto, quantity)
    error_msg_convertto = "Unrecognized unit system '%s'." % convertto
    
    if convertto == 'PPMUnits' and fromtype == 'mesaprof':
        m = ms.mesa_profile(datadir, modordump)
        if quantity == 'density':
            inarray = 10**m.get('logRho')
            print("Converting from g/cm**3 to kg/cm**3")  
        elif quantity == 'pressure':
            inarray = m.get('pressure')
            print("Converting from barye to 10**19 barye")
        elif quantity == 'temperature':
            print("Converting from K to 10**9 K")
            inarray = m.get('temperature')
        elif quantity == 'radius':
            print("Converting from R_sun to Mm")
            inarray = m.get('radius')
        else:
            raise NotImplementedError(error_msg_quantity)
    
        return(inarray / to_cgs[quantity])
    
    
    elif convertto == 'MESAUnits' and fromtype == 'rprof' :
        l = ppm.RprofSet(datadir)
        if quantity == 'density':
            inarray = l.get('Rho0',fname=modordump, num_type='NDump', resolution='h')[::2]\
            +l.get('Rho1',fname=modordump, num_type='NDump', resolution='h')[::2]
            print("Converting from kg/cm**3 to g/cm**3")  
        elif quantity == 'pressure':
            print("Converting from 10**19 barye to barye")
            inarray = l.get('P0',fname=modordump, num_type='NDump', resolution='h')[::2]\
            +l.get('P1',fname=modordump, num_type='NDump', resolution='h')[::2]
        elif quantity == 'temperature':
            print("Converting from 10**9 K to K")
            inarray = l.get('T9', fname=modordump, num_type='NDump', resolution='l')
        elif quantity == 'radius':
            print("Converting from Mm to R_sun")
            inarray = l.get('R', fname=modordump, num_type='NDump', resolution='l')
        else:
            raise NotImplementedError(error_msg_quantity)
            
        return(inarray * to_cgs[quantity])
        
        
    elif convertto == 'MESAUnits' and fromtype == 'ppmsetup':
        data = ascii_table(sldir=datadir, filename=filename)
        if quantity == 'density':
            inarray = data.get('rho')
            print("Converting from kg/cm**3 to g/cm**3")  
        elif quantity == 'pressure':
            print("Converting from 10**19 barye to barye")
            inarray = data.get('P')
        elif quantity == 'temperature':
            print("Converting from 10**9 K to K")
            inarray = data.get('T')
        elif quantity == 'radius':
            print("Converting from Mm to R_sun")
            inarray = data.get('radius')
        else:
            raise NotImplementedError(error_msg_quantity)
            
        return(inarray * to_cgs[quantity])
    else:
        raise NotImplementedError(error_msg_convertto)
        
def eosPS(rho, T, mu, units, tocompute='S', idealgas=False):
    '''
    using density, temperature and mean molecular weight calculate entropy or pressure 
    profile for either ppm or mesa data

    units, str: Which units to compute in options are MESA or PPM
    rho, T, mu: 1D arrays
    tocompute, str: what you want out, options are S or P
    idealgas, boolean
    '''
    if idealgas == False:
        f = 1
    else:
        f = 0

    if units == 'MESA':
        Rgas = R_cgs
        ac   = a_cgs

    if units == 'PPM':
        Rgas = R_code
        ac   = a_code
    
    if tocompute == 'S':
        outarray = (3./2.)*(Rgas/mu)*np.log(T) - (Rgas/mu)*np.log(rho) + f*((4./3.)*(ac*T**3) / (rho))
    if tocompute == 'P':
        outarray = (Rgas/mu)*rho*T + f*((ac/3)*T**4)
    return(outarray)
        
def EOSgasrad(T,rho,mu,a,R):
    '''S and P for rad and gas
    Takes also a and R as input, so can be used for both
    mesa and code units'''
    Rbymu = R/mu
    S = (3./2.)*Rbymu * np.log(T) - Rbymu*np.log(rho) + (4./3)*a*T**3/rho
    P = Rbymu*rho*T +(1./3)*a*T**4
    return np.array([S,P])

def rhoTfromSP(T,rho,S,P,a,R,mu):
    '''EOS inversion, S, P in, rho, T out

    equations to solve
    dG/dx * delta = -G(x)
    here two equations: J(rhoT) * delta_rhoT = -GSP(rhoT) [=G]
    where rhoT = (rho,T), and (G1,G2) = GSP(rhoT) 
    J11 * drho + J21 * dT = -G1
    J12 * drho + J22 * dT = -G2

    Note: An easier solution would have been to solve P(rho,T) for 
          rho and just insert into S(rho,T) and then solve for T
          iteratively, as Huaqing pointed out ;-)
    '''
    eps_g = 1.e-7; eps = 1.
    [G1,G2] = EOSgasrad(T,rho,mu,a,R)-np.array([S,P]) 
    while eps > eps_g:
        Rbymu = R/mu
        J11 = -Rbymu/rho - (4./3.)*a*T**3./rho**2      #dG1/drho
        J12 = Rbymu*T                                  #dG2/drho
        J21 = 3./2.* Rbymu/T + 4*a*T**2/rho            #dG1/dT
        J22 = Rbymu*rho + 4./3.*(a*T**3)               #dG2/dT
        dT = (- G1*J12/J11 + G2) / (J12*J21/J11 - J22)
        drho  = (-G2 - J22 * dT) / J12 
        T = T + dT
        rho = rho + drho
        [G1,G2] = EOSgasrad(T,rho,mu,a,R)-np.array([S,P]) 
        eps = max(np.array([G1,G2]+[drho,dT]))       
    return rho, T

def rhs4(x,r,T,rho,S,P,a_code,R_code,mu):
    '''RHS of ODE dP/dr and dm/dr using rho, T fron NR.'''
    m,P = x
    rho,T = rhoTfromSP(T,rho,S,P,a_code,R_code,mu)
    f1 = 4.*np.pi*r**2*rho
    if r == 0:    # for integration from zero
        f2 = 0
    else:
        f2 = -G_code*m*rho/r**2
    return [f1,f2]


def get_prof_data(data_dir,model):
    '''
    This function returns the mesa profile data needed for a ppm setup. It must and only returns what is needed in the order specified. 
    usage : log_conv_vel, radius_mesa, P_mesa, T_mesa, entropy_mesa, rho_mesa, mass_mesa, mu_mesa ,\
pgas_div_ptotal = ps.get_prof_data(ddir,model)
    '''
    mprof=ms.mesa_profile(data_dir,num=model)
    print("returning log_conv_vel, radius_mesa, P_mesa, T_mesa, entropy_mesa, rho_mesa, mass_mesa, mu_mesa and pgas_div_ptotal")
    log_conv_vel=mprof.get('log_conv_vel')
    radius_mesa = mprof.get('radius')
    P_mesa = mprof.get('pressure')
    T_mesa = mprof.get('temperature')
    entropy_mesa = mprof.get('entropy')
    rho_mesa = 10**mprof.get('logRho')
    mass_mesa = mprof.get('mass')
    mu_mesa = mprof.get('mu')
    pgas_div_ptotal = mprof.get('pgas_div_ptotal')
    return(log_conv_vel, radius_mesa, P_mesa, T_mesa, entropy_mesa, rho_mesa, mass_mesa, mu_mesa , pgas_div_ptotal)

    
    
def mufromX(X, A, Z):
    ''' Input is list of list of abundances, list of mass numbers and list of charge numbers.
    e.g. abus = [[0.75,0.75,0.75], [0.24,0.24,0.24], [0.01,0.01,0.01]]
    A = [1, 4, 12] 
    Z = [1, 2, 6] 
    '''
    
    x = np.array(X).T
    a = np.array(A)
    z = np.array(Z)
    mu_e_recip = np.zeros(len(x))
    mu_i_recip = np.zeros(len(x))
    for i in range(len(x)):
        mu_e_recip[i] = 1/np.sum(x[i]*z/a)
        mu_i_recip[i] = 1/np.sum(x[i]/a)   
    mu_tot = 1/(1/mu_e_recip+ 1/mu_i_recip)
    return(mu_tot)


def get_burning_coeffs(mesa_prof, r0, iso_fuel):
    stable_isotopes = (\
    'h1   ','h2   ','he3  ','he4  ','li6  ','li7  ','be9  ','b10  ', \
    'b11  ','c12  ','c13  ','n14  ','n15  ','o16  ','o17  ','o18  ', \
    'f19  ','ne20 ','ne21 ','ne22 ','na23 ','mg24 ','mg25 ','mg26 ', \
    'al27 ','si28 ','si29 ','si30 ','p31  ','s32  ','s33  ','s34  ', \
    's36  ','cl35 ','cl37 ','ar36 ','ar38 ','ar40 ','k39  ','k40  ', \
    'k41  ','ca40 ','ca42 ','ca43 ','ca44 ','ca46 ','ca48 ','sc45 ', \
    'ti46 ','ti47 ','ti48 ','ti49 ','ti50 ','v50  ','v51  ','cr50 ', \
    'cr52 ','cr53 ','cr54 ','mn55 ','fe54 ','fe56 ','fe57 ','fe58 ', \
    'co59 ','ni58 ','ni60 ','ni61 ','ni62 ','ni64 ','cu63 ','cu65 ', \
    'zn64 ','zn66 ','zn67 ','zn68 ','zn70 ','ga69 ','ga71 ','ge70 ', \
    'ge72 ','ge73 ','ge74 ','ge76 ','as75 ','se74 ','se76 ','se77 ', \
    'se78 ','se80 ','se82 ','br79 ','br81 ','kr78 ','kr80 ','kr82 ', \
    'kr83 ','kr84 ','kr86 ','rb85 ','rb87 ','sr84 ','sr86 ','sr87 ', \
    'sr88 ','y89  ','zr90 ','zr91 ','zr92 ','zr94 ','zr96 ','nb93 ', \
    'mo92 ','mo94 ','mo95 ','mo96 ','mo97 ','mo98 ','mo100','ru96 ', \
    'ru98 ','ru99 ','ru100','ru101','ru102','ru104','rh103','pd102', \
    'pd104','pd105','pd106','pd108','pd110','ag107','ag109','cd106', \
    'cd108','cd110','cd111','cd112','cd113','cd114','cd116','in113', \
    'in115','sn112','sn114','sn115','sn116','sn117','sn118','sn119', \
    'sn120','sn122','sn124','sb121','sb123','te120','te122','te123', \
    'te124','te125','te126','te128','te130','i127 ','xe124','xe126', \
    'xe128','xe129','xe130','xe131','xe132','xe134','xe136','cs133', \
    'ba130','ba132','ba134','ba135','ba136','ba137','ba138','la138', \
    'la139','ce136','ce138','ce140','ce142','pr141','nd142','nd143', \
    'nd144','nd145','nd146','nd148','nd150','sm144','sm147','sm148', \
    'sm149','sm150','sm152','sm154','eu151','eu153','gd152','gd154', \
    'gd155','gd156','gd157','gd158','gd160','tb159','dy156','dy158', \
    'dy160','dy161','dy162','dy163','dy164','ho165','er162','er164', \
    'er166','er167','er168','er170','tm169','yb168','yb170','yb171', \
    'yb172','yb173','yb174','yb176','lu175','lu176','hf174','hf176', \
    'hf177','hf178','hf179','hf180','ta180','ta181','w180 ','w182 ', \
    'w183 ','w184 ','w186 ','re185','re187','os184','os186','os187', \
    'os188','os189','os190','os192','ir191','ir193','pt190','pt192', \
    'pt194','pt195','pt196','pt198','au197','hg196','hg198','hg199', \
    'hg200','hg201','hg202','hg204','tl203','tl205','pb204','pb206', \
    'pb207','pb208','bi209','th232','u235 ','u238 ')
    stable_isotopes = [iso.strip() for iso in stable_isotopes]

    r = (nugridpy.constants.r_sun/1.e8)*mesa_prof.get('radius')
    idx0 = np.argmin(np.abs(r - r0))
    r0 = r[idx0]

    # Average mass number.
    avg_A = 0.
    Y_fluid = 0.
    X_lim = 1e-3
    print('Mass fractions of species with X > {:.1e} at r = {:.3f} Mm:'.\
          format(X_lim, r0))
    for iso in stable_isotopes:
        if iso in mesa_prof.cols:
            # iso's mass fraction
            X_iso = mesa_prof.get(iso)[idx0]

            # iso's mass number
            A_iso = float(re.findall(r'\d+', iso)[0])

            # We want to weight the average A by the number of nuclei.
            # X_iso*dm is the mass of iso in the mixture of mass dm.
            # X_iso*dm/(A_iso*amu), where amu is the atomic mass unit,
            # is the number of nuclei of isotope iso in dm. The number
            # fraction Y is defined for dm = amu.
            Y_iso = X_iso/A_iso
            Y_fluid += Y_iso

            if X_iso > X_lim:
                print(iso, X_iso)

    # Y_fluid = X_fluid/avg_A; X_fluid = 1.
    avg_A = 1./Y_fluid
    print('Average mass number: {:.6f}'.format(avg_A))
    
    X_iso_fuel = mesa_prof.get(iso_fuel)[idx0]
    A_iso_fuel = float(re.findall(r'\d+', iso_fuel)[0])
    Y_iso_fuel = X_iso_fuel/A_iso_fuel
    fk = Y_iso_fuel/Y_fluid
    print('Number fraction of fuel ({:s}): {:.6f}\n'.format(iso_fuel, fk))
    
    return avg_A, fk
