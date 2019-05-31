'''pm setup tools

This module provides a number of utilities that support the 
construction of a new setup for the PPMstar code. 

FH, 20140907

OC, 20180414 , 20190211
'''
from nugridpy import mesa as ms
from nugridpy import astronomy as ast
from ppmpy import ppm
import numpy as np
import sys
from nugridpy.ascii_table import ascii_table 

G_code = ast.grav_const*1000
a_cgs  = ast.radiation_constant
a_code = a_cgs * 10**17
R_cgs  = ast.boltzmann_constant*ast.avogadro_constant
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
    Rgasconst = ast.boltzmann_constant*ast.avogadro_constant/1e7
    amuairbyR = airmu / Rgasconst
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
    
    Rgasconst = ast.boltzmann_constant*ast.avogadro_constant/1e7 
    mubyR = mu / Rgasconst
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
