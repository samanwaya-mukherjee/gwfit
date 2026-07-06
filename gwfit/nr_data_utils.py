import sxs
import numpy as np 
import h5py as h5
import useful_relations as urel 
from scipy.interpolate import interp1d


def nr_resampled(path_to_file,Ms,DL=200,merger_time=None,time_fix=False,fine_resample=False,geo_units=False):
    

    M = Ms*urel.MTsun
    MbyDL = (Ms/DL)*urel.strain_fac

    # f = h5.File(path_to_file,'r')
    # dataset = f['Extrapolated_N2.dir']
    # data = dataset['Y_l2_m2.dat']

    with h5.File(path_to_file, "r") as f:
        data = f["Extrapolated_N2.dir"]["Y_l2_m2.dat"][:]

    time0 = data[:,0]
    strainp0 = data[:,1]
    strainc0 = -data[:,2]

    time = time0*M
    strainp, strainc = strainp0*MbyDL, strainc0*MbyDL

    
    ### DEFINING THE CHIRP TIME AS 0 BY FINDING THE MAXIMUM STRAIN ####
    if time_fix==True:
        if merger_time is None:
            j = np.argmax(strainp)
            maximt = time[j]

        else: maximt = merger_time*M

        time = time - maximt
        
    flinp = interp1d(time,strainp,kind='cubic')
    flinc = interp1d(time,strainc,kind='cubic')


    # finding maximum and minimum time spacing ---------
    deltat=[]
    for i in range(len(time)-1):
        deltat.append(time[i+1]-time[i])

    deltat.append(deltat[-1])

    coarse_dt, fine_dt = max(deltat), min(deltat)
    #--------------------------------------------------

    if fine_resample == True: 
        dt = fine_dt
    else: dt = coarse_dt

    time_resamp = np.arange(time[0],time[-1],dt)
    strainp_resamp = flinp(time_resamp)
    strainc_resamp = flinc(time_resamp)

    if geo_units == True: return time0, strainp0, strainc0, time_resamp, strainp_resamp, strainc_resamp
    return time_resamp, strainp_resamp, strainc_resamp

def nr_resampled_geo(path_to_file,merger_time=None,fine_resample=False):
   
    # f = h5.File(path_to_file,'r')
    # dataset = f['Extrapolated_N2.dir']
    # data = dataset['Y_l2_m2.dat']

    with h5.File(path_to_file, "r") as f:
        data = f["Extrapolated_N2.dir"]["Y_l2_m2.dat"][:]

    time = data[:,0]
    strainp = data[:,1]
    strainc = -data[:,2]

       
    ### DEFINING THE CHIRP TIME AS 0 BY FINDING THE MAXIMUM STRAIN ####

    if merger_time is None:
        
        maximstr = strainp[0]

        for i in range(len(time)):
            if strainp[i] > maximstr: 
                maximstr = strainp[i]
                j=i
        maximt = time[j]
        
    else: maximt = merger_time


    # for k in range(len(time)):
    #     time[k]=time[k]-maximt
        
    flinp = interp1d(time,strainp,kind='cubic')
    flinc = interp1d(time,strainc,kind='cubic')


    # finding maximum and minimum time spacing ---------
    deltat=[]
    for i in range(len(time)-1):
        deltat.append(time[i+1]-time[i])

    deltat.append(deltat[-1])

    coarse_dt, fine_dt = max(deltat), min(deltat)
    #--------------------------------------------------

    if fine_resample == True: 
        dt = fine_dt
    else: dt = coarse_dt

    time_resamp = np.arange(time[0],time[-1],dt)
    strainp_resamp = flinp(time_resamp)
    strainc_resamp = flinc(time_resamp)

    return time_resamp, strainp_resamp, strainc_resamp


def nr_resampled_rit(path_to_file,Ms,DL=200,geo_units=False):
    

    M = Ms*urel.MTsun
    MbyDL = (Ms/DL)*urel.strain_fac

    file = h5.File(path_to_file,'r')
    phasedataset=file['phase_l2_m2']
    phasex=phasedataset['X'][:,]
    phasey=phasedataset['Y'][:,]
    ampdataset=file['amp_l2_m2']
    ampx=ampdataset['X'][:,]
    ampy=ampdataset['Y'][:,]
    timedata=file['NRTimes'][:,]

    phaseint = interp1d(phasex,phasey,kind='cubic')
    phase = phaseint(timedata)

    ampint = interp1d(ampx,ampy,kind='cubic')
    amp = ampint(timedata)

    sp0 = amp*np.cos(phase)
    sc0 = amp*np.sin(phase)
    
    time = timedata*M
    sp, sc = sp0*MbyDL, sc0*MbyDL
    
    if geo_units==True: return timedata, sp0, sc0, time, sp , sc
    else: return time, sp, sc
    
def nr_resampled_horizon(path_to_file,quantity=None,delta_tbyM=None,relaxtime=None):
    
    if quantity is None: raise Exception('Provide the quantity you need. Allowed entries are: ChristodoulouMass, ArealMass, CoordCenterInertial, DimensionfulInertialSpin, DimensionfulInertialSpinMag, chiInertial, chiMagInertial')
    if delta_tbyM is None: raise Exception('Provide delta_tbyM')
    # if relaxtime is None: raise Exception('Provide relaxtime')
    file2 = h5.File(path_to_file,'r')
    datasetA=file2['AhA.dir']
    dataA=datasetA['{}.dat'.format(quantity)]
    datasetB=file2['AhB.dir']
    dataB=datasetB['{}.dat'.format(quantity)]
    time=dataA[:,0]
    # index1 = np.where(time>relaxtime)[0][0]
    # time2 = time[index1:]
    A=dataA[:,1]
    # A2=A[index1:]
    B=dataB[:,1]
    # B2=B[index1:]
    
    flinA = interp1d(time,A,kind='cubic')
    flinB = interp1d(time,B,kind='cubic')
    
    time_resamp = np.arange(time[0],time[-1],delta_tbyM)
    A_resamp = flinA(time_resamp)
    B_resamp = flinB(time_resamp)
    
    return time_resamp, A_resamp, B_resamp

def nr_resampled_horizon_with_sxs(simulation_label='SXS:BBH:0001',sampling='coarse',delta_t=None,geo_units=True):
    
    simulation = sxs.load(simulation_label)
    horizons = simulation.horizons
    metadata = simulation.metadata
    reference_time = metadata.reference_time

    reference_index = np.searchsorted(horizons.A.time, reference_time)

    time_sliced = horizons.A.time[reference_index:]
    mass_sliced_A = horizons.A.christodoulou_mass[reference_index:]
    mass_sliced_B = horizons.B.christodoulou_mass[reference_index:]
    
    if geo_units:
        time = time_sliced
        massA = mass_sliced_A
        massB = mass_sliced_B
    else:
        time = time_sliced * urel.MTsun
        massA = mass_sliced_A * urel.MTsun
        massB = mass_sliced_B * urel.MTsun

    flinA = interp1d(time,massA,kind='cubic')
    flinB = interp1d(time,massB,kind='cubic')

    if delta_t is None:
        if sampling == "fine":
            delta_t = np.diff(time).min()
        elif sampling == "coarse":
            delta_t = np.diff(time).max()
        else:
            raise ValueError("sampling must be 'coarse' or 'fine'")

    time_sliced_even = np.arange(time[0],time[-1],delta_t)

    mass_A_even = flinA(time_sliced_even)
    mass_B_even = flinB(time_sliced_even)

    return time_sliced_even, mass_A_even, mass_B_even


def nr_resampled_geo_with_sxs(simulation_label='SXS:BBH:0001',mode=[2,2],sampling=None,delta_t=None,junk_remove=True,merger_cutoff=False):
    '''
    simulation_label => just the 4-digit sxs label. 
    mode = an array of two elements: [l,m]
    sampling => if none is specified, the resampling will be with the coarsest delta_t (sometimes it's enough). Otherwise, you can pass the string 'fine' to resample with the finest delta_t.
    delta_t => you can choose delta_t of your own (optional)
    junk_remove => trim the starting part of the waveform up to the reference time to remove junk.
    merger_cutoff => cutoff the waveform at the common horizon time.

    Returns
         resampled (l,m) mode: time, real and -imag parts.
    '''
    simulation = sxs.load(simulation_label)
    ell, em = mode
    n = ell+ em
    
    for i in range(2,ell): 
        n += 2*i+1
        
    h = simulation.h[:,n]  #for the (l,m) mode
    metadata = simulation.metadata
    reference_time = metadata.reference_time
    merger_time = metadata.common_horizon_time
    
    reference_index = np.searchsorted(h.t, reference_time)

    start = reference_index if junk_remove else 0
    stop = np.searchsorted(h.t, merger_time) if merger_cutoff else None

    time = h.t[start:stop]
    strainp = h.real[start:stop]
    strainc = -h.imag[start:stop]
        
    flinp = interp1d(time,strainp,kind='cubic')
    flinc = interp1d(time,strainc,kind='cubic')

    dt_array = np.diff(time)
    if delta_t is None:
        if sampling == "fine":
            dt = dt_array.min()
        else:
            dt = dt_array.max()
    else:
        dt = delta_t

    time_resamp = np.arange(time[0],time[-1],dt)
    strainp_resamp = flinp(time_resamp)
    strainc_resamp = flinc(time_resamp)

    return time_resamp, strainp_resamp, strainc_resamp

def gw_flux(simulation_label='SXS:BBH:0001',merger_cutoff=True,delta_t=None,ell_max=8):
    i=0
    M = urel.MTsun
    for ell in range(2,ell_max+1):
        for em in range(-ell,ell+1):
            t, hlm_real, hlm_imag = nr_resampled_geo_with_sxs(simulation_label=simulation_label,mode=[ell,em],merger_cutoff=merger_cutoff,delta_t=delta_t)
            if i==0: 
                gw_flux = np.zeros_like(t)
                t_nr = t
            hlm_real_dot = urel.dydx(t*M,hlm_real)
            hlm_imag_dot = urel.dydx(t*M,hlm_imag)
            
            hlm_real_dot *= M
            hlm_imag_dot *= M
            
            gw_flux += np.abs(hlm_real_dot**2+hlm_imag_dot**2)
            i += 1
        
    gw_flux *= (1./(16.*np.pi))
    
    return t_nr, gw_flux
  
