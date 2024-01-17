import numpy as np
from scipy import constants as const
from astropy import constants
import cantera as ct
from scipy import integrate
import pickle

from photochem import Atmosphere, PhotoException
from photochem.clima import AdiabatClimate
import utils
import planets
from photochem.utils._format import FormatSettings_main, MyDumper, Loader, yaml


class TempPress:

    def __init__(self, P, T):
        self.log10P = np.log10(P)[::-1] # P in dynes/cm^2
        self.T = T[::-1] # T in K

    def temperature(self, P):
        return np.interp(np.log10(P), self.log10P, self.T)
    
def gravity(radius, mass, z):
    "CGS units"
    G_grav = const.G
    grav = G_grav * (mass/1.0e3) / ((radius + z)/1.0e2)**2.0
    grav = grav*1.0e2 # convert to cgs
    return grav # cm/s^2

def rhs_alt(P, u, mubar, radius, mass, pt):

    # Currently altitude in cm
    z = u[0]

    # compute gravity
    grav = gravity(radius, mass, z)

    # interpolate P-T profile
    T = pt.temperature(P)

    k_boltz = const.Boltzmann*1e7
    dz_dP = -(k_boltz*T*const.Avogadro)/(mubar*grav*P)

    return np.array([dz_dP])

def chemical_equilibrium_PT(P, T, ct_file, atoms, M_H_metalicity, CtoO):
    '''Given a P-T profile and metalicity, this function computes chemical
    chemical equilibrium for the entire atmospheric column. CGS units.
    '''

    comp = utils.composition_from_metalicity_for_atoms(atoms, M_H_metalicity)

    # Adjust C and O to get desired C/O ratio. CtoO is relative to solar
    x = CtoO*(comp['C']/comp['O'])
    a = (x*comp['O'] - comp['C'])/(1+x)
    comp['C'] = comp['C'] + a
    comp['O'] = comp['O'] - a
    
    gas = ct.Solution(ct_file)

    mubar = np.empty(P.shape[0])
    equi = {}
    for i,sp in enumerate(gas.species_names):
        equi[sp] = np.empty(P.shape[0])
    for i in range(P.shape[0]):
        gas.TPX = T[i],P[i]/10,comp
        gas.equilibrate('TP')
        mubar[i] = gas.mean_molecular_weight
        for j,sp in enumerate(gas.species_names):
            equi[sp][i] = gas.X[j]

    surf = {}
    for i,sp in enumerate(gas.species_names):
        surf[sp] = equi[sp][0]

    return equi, surf, mubar

def altitude_profile_PT(P, T, radius, mass, mubar):
    '''Computes altitude given P-T. CGS units.
    '''
    pt = TempPress(P, T)
    args = (mubar, radius, mass, pt)
    z0 = 0.0
    
    out = integrate.solve_ivp(rhs_alt, [P[0], P[-1]], np.array([z0]), t_eval=P, args=args, rtol=1e-5)

    z = out.y[0]

    return z

def write_atmosphere_file(filename, alt, press, den, temp, eddy, mix):

    fmt = '{:25}'
    with open(filename, 'w') as f:
        f.write(fmt.format('alt'))
        f.write(fmt.format('press'))
        f.write(fmt.format('den'))
        f.write(fmt.format('temp'))
        f.write(fmt.format('eddy'))

        for key in mix:
            f.write(fmt.format(key))
        
        f.write('\n')

        for i in range(press.shape[0]):
            f.write(fmt.format('%e'%alt[i]))
            f.write(fmt.format('%e'%press[i]))
            f.write(fmt.format('%e'%den[i]))
            f.write(fmt.format('%e'%temp[i]))
            f.write(fmt.format('%e'%eddy[i]))

            for key in mix:
                f.write(fmt.format('%e'%mix[key][i]))

            f.write('\n')

def surf_boundary_conditions(surf, min_mix, sp_to_exclude):
    bc_list = []
    for sp in surf:
        if surf[sp] > min_mix and sp not in sp_to_exclude:
            lb = {"type": "mix", "mix": float(surf[sp])}
            ub = {"type": "veff", "veff": 0.0}
            entry = {}
            entry['name'] = sp
            entry['lower-boundary'] = lb
            entry['upper-boundary'] = ub
            bc_list.append(entry)
    out = {}
    out['boundary-conditions'] = bc_list

    return out

def write_quench_settings_file(settings_in, settings_out, surf, min_mix, sp_to_exclude, top, nz, P_surf):

    fil = open(settings_in,'r')
    settings = yaml.load(fil,Loader=Loader)
    fil.close()

    settings['atmosphere-grid']['top'] = float(top)
    settings['atmosphere-grid']['number-of-layers'] = int(nz)
    settings['planet']['surface-pressure'] = float(P_surf/1e6)

    # boundary conditions
    bc = surf_boundary_conditions(surf, min_mix, sp_to_exclude)
    settings['boundary-conditions'] = bc['boundary-conditions']

    out = FormatSettings_main(settings)
    with open(settings_out,'w') as f:
        yaml.dump(out, f, Dumper=MyDumper ,sort_keys=False, width=70)

def write_quench_files(settings_in, settings_out, atmosphere_out, P, T, M_H_metalicity, CtoO, ct_file, atoms, min_mix, nz, eddy):

    radius = planets.k2_18b.radius*(constants.R_earth.value)*1e2
    mass = planets.k2_18b.mass*(constants.M_earth.value)*1e3

    equi, surf, mubar = chemical_equilibrium_PT(P, T, ct_file, atoms, M_H_metalicity, CtoO)
    z = altitude_profile_PT(P, T, radius, mass, mubar[0])
    write_quench_settings_file(settings_in, settings_out, surf, min_mix, ['H2'], z[-1], nz, P[0])

    alt = z/1e5 # to km
    press = P/1e6 # to bar
    den = P/(const.Boltzmann*1e7*T)
    temp = T
    eddy_ = np.ones(P.shape[0])*eddy
    write_atmosphere_file(atmosphere_out, alt, press, den, temp, eddy_, equi)

def P_T_from_file(filename, P_bottom, P_top):
    nz = 100

    with open(filename,'rb') as f:
        out = pickle.load(f)
    
    P_c = out['pressure'].copy()[::-1]*1e6
    T_c = out['temperature'].copy()[::-1]

    if P_bottom > P_c[0]:
        raise Exception()

    if P_top < P_c[-1]:
        P_c = np.append(P_c, P_top)
        T_c = np.append(T_c, T_c[-1])

    P = np.logspace(np.log10(P_bottom), np.log10(P_top), nz)

    T = np.interp(np.log10(P).copy()[::-1], np.log10(P_c).copy()[::-1], T_c.copy()[::-1])
    T = T.copy()[::-1]

    return P, T

def integrate_quench_equilibrium(pc, P, T, P_top):
    pc.var.verbose = 0
    pc.var.atol = 1e-25
    pc.var.rtol = 1e-5
    pc.initialize_stepper(pc.wrk.usol)

    usol_prev = pc.wrk.usol.copy()
    counter = 0
    nerrors = 0
    try:
        while True: 
            try:
                for i in range(500):
                    pc.step()
            except PhotoException as e:
                usol = np.clip(pc.wrk.usol,a_min=1.0e-40,a_max=np.inf)
                pc.initialize_stepper(usol)
                if nerrors > 10:
                    raise PhotoException(e)
                nerrors += 1
            
            usol_new = pc.wrk.usol.copy()
            inds = np.where(usol_new>1e-10)
            rel_change = np.max(np.abs(usol_new[inds]/usol_prev[inds] - 1))
            print('%.2e  %i'%(rel_change, counter))
            usol_prev = usol_new.copy()
            if rel_change < 1e-3:
                break
            if counter > 20:
                pc.update_vertical_grid(TOA_pressure=P_top)
                pc.set_press_temp_edd(P.copy(), T.copy(), (np.ones(T.shape[0])*pc.var.edd[0]).copy())
                pc.initialize_stepper(pc.wrk.usol)
                counter = 0
            counter += 1
    except KeyboardInterrupt:
        # Manually stop integration, if desired.
        pass

def make_clima_profile_from_quench(c, pc, T_trop, P_top):
    
    surf = {}
    for sp in pc.dat.species_names[:-2]:
        ind = pc.dat.species_names.index(sp)
        tmp = pc.wrk.densities[ind,-1]/pc.wrk.density[-1]
        surf[sp] = tmp

    f_i = np.empty(len(c.species_names))
    for i,sp in enumerate(c.species_names):
        f_i[i] = np.maximum(surf[sp],1e-40)

    P_surf = pc.wrk.pressure[-1]
    P_i = f_i*P_surf
    bg_gas = 'H2'
    c.T_trop = T_trop
    c.solve_for_T_trop = False
    c.RH = np.ones(len(c.species_names))
    c.P_top = P_top
    T_surf = pc.var.temperature[-1]
    c.make_profile_bg_gas(T_surf, P_i, P_surf, bg_gas)

    # Find pressure where H2O condenses
    ind = c.species_names.index('H2O')
    assert c.f_i[0,ind] == c.f_i[1,ind]

    for i in range(c.f_i[:,ind].shape[0]):
        if c.f_i[i,ind] < c.f_i[0,ind]:
            ind1 = i
            P_condense = c.P[ind1]
            break

    P_trop = c.P_trop

    return surf, P_condense, P_trop

def write_photochem_settings_file(settings_in, settings_out, surf, min_mix, sp_to_exclude, top, P_surf, P_condense, P_trop):

    fil = open(settings_in,'r')
    settings = yaml.load(fil,Loader=Loader)
    fil.close()

    settings['atmosphere-grid']['top'] = float(top)
    settings['planet']['surface-pressure'] = float(P_surf/1e6)

    # boundary conditions
    bc = surf_boundary_conditions(surf, min_mix, sp_to_exclude)
    settings['boundary-conditions'] = bc['boundary-conditions']

    out = FormatSettings_main(settings)
    # tack on some info about clouds
    out['clouds'] = {}
    out['clouds']['P-condense'] = float(P_condense)
    out['clouds']['P-trop'] = float(P_trop)
    with open(settings_out,'w') as f:
        yaml.dump(out, f, Dumper=MyDumper ,sort_keys=False, width=70)

def make_picaso_input_neptune(outfile):
    pc1 = Atmosphere('input/zahnle_earth_new_noparticles.yaml',\
                    outfile+'_settings_quench.yaml',\
                    "input/k2_18b_stellar_flux.txt",\
                    outfile+'_atmosphere_quench_c.txt',
                    data_dir='/Users/nicholas/Documents/Research_local/PhotochemPy/photochem_clima_data')
    pc2 = Atmosphere('input/zahnle_earth_new_S8.yaml',\
                    outfile+'_settings_photochem.yaml',\
                    "input/k2_18b_stellar_flux.txt",\
                    outfile+'_atmosphere_photochem_c.txt',
                    data_dir='/Users/nicholas/Documents/Research_local/PhotochemPy/photochem_clima_data')
    
    mix = {}
    mix['press'] = np.append(pc1.wrk.pressure,pc2.wrk.pressure)
    mix['temp'] = np.append(pc1.var.temperature,pc2.var.temperature)
    for i,sp in enumerate(pc1.dat.species_names[pc1.dat.np:-2]):
        ind = pc1.dat.species_names.index(sp)
        tmp1 = pc1.wrk.densities[ind,:]/pc1.wrk.density

        ind = pc2.dat.species_names.index(sp)
        tmp2 = pc2.wrk.densities[ind,:]/pc2.wrk.density
        
        mix[sp] = np.append(tmp1,tmp2)

    species = pc1.dat.species_names[pc1.dat.np:-2]
    utils.write_picaso_atmosphere(mix, outfile+'_picaso.pt', species)
    

def run_quench_photochem_model(settings_quench_in, settings_photochem_in, PTfile_in, outfile, P_bottom, P_top, M_H_metalicity, 
                               CtoO, ct_file, atoms, min_mix, nz_q, eddy_q,
                               T_trop, P_top_clima, eddy_p, equilibrium_time):
                               
    settings_quench_out = outfile+"_settings_quench.yaml"
    settings_photochem_out = outfile+"_settings_photochem.yaml"
    atmosphere_quench_out = outfile+"_atmosphere_quench.txt"
    atmosphere_photochem_out = outfile+"_atmosphere_photochem.txt"

    P, T = P_T_from_file(PTfile_in, P_bottom, P_top)
    write_quench_files(settings_quench_in, settings_quench_out, atmosphere_quench_out, P, T, M_H_metalicity, CtoO, ct_file, atoms, min_mix, nz_q, eddy_q)

    pc_q = Atmosphere('input/zahnle_earth_new_noparticles.yaml',\
                    settings_quench_out,\
                    "input/k2_18b_stellar_flux.txt",\
                    atmosphere_quench_out,
                    data_dir='/Users/nicholas/Documents/Research_local/PhotochemPy/photochem_clima_data')
    pc_q.var.custom_binary_diffusion_fcn = utils.custom_binary_diffusion_fcn
    integrate_quench_equilibrium(pc_q, P, T, P_top)

    atmosphere_out_c = outfile+"_atmosphere_quench_c.txt"
    pc_q.out2atmosphere_txt(atmosphere_out_c, overwrite=True)

    with open(settings_quench_out,'r') as f:
        settings = yaml.load(f,Loader=Loader)
    settings['atmosphere-grid']['top'] = float(pc_q.var.top_atmos)
    settings = FormatSettings_main(settings)
    with open(settings_quench_out,'w') as f:
        yaml.dump(settings, f, Dumper=MyDumper ,sort_keys=False, width=70)

    c = AdiabatClimate('input/neptune/species_quench_climate.yaml',
                       'input/neptune/settings_quench_climate.yaml',
                       'input/k2_18b_stellar_flux.txt')
    
    surf, P_condense, P_trop = make_clima_profile_from_quench(c, pc_q, T_trop, P_top_clima)

    settings_in = 'input/neptune/settings_neptune_photochem_template.yaml'
    min_mix_photochem = 1e-20
    sp_to_exclude = ['H2']
    write_photochem_settings_file(settings_photochem_in, settings_photochem_out, surf, min_mix_photochem, sp_to_exclude, c.z[-1], c.P_surf, P_condense, P_trop)

    log10P_trop = np.log10(c.P_trop/1e6)
    log10P = np.log10(c.P/1e6)
    Kzz_trop = eddy_p
    eddy_ = utils.simple_eddy_diffusion_profile(log10P, log10P_trop, Kzz_trop)
    c.out2atmosphere_txt(atmosphere_photochem_out, eddy_, overwrite=True)

    pc = Atmosphere('input/zahnle_earth_new_S8.yaml',\
                    settings_photochem_out,\
                    "input/k2_18b_stellar_flux.txt",\
                    atmosphere_photochem_out,
                    data_dir='/Users/nicholas/Documents/Research_local/PhotochemPy/photochem_clima_data')
    pc.var.custom_binary_diffusion_fcn = utils.custom_binary_diffusion_fcn
    pc.var.equilibrium_time = equilibrium_time
    pc.var.atol = 1e-25
    pc.var.rtol = 1e-3
    pc.initialize_stepper(pc.wrk.usol)
    tn = 0.0
    counter = 0
    try:
        while tn < pc.var.equilibrium_time:
            tn = pc.step()
            counter += 1
            if counter > 3000:
                pc.update_vertical_grid(TOA_pressure=1e-8*1e6)
                pc.set_press_temp_edd(c.P, c.T, eddy_, c.P_trop)
                pc.initialize_stepper(pc.wrk.usol)
                counter = 0
    except KeyboardInterrupt:
        # Manually stop integration, if desired.
        pass

    # save result
    atmosphere_out_c = outfile+"_atmosphere_photochem_c.txt"
    pc.out2atmosphere_txt(atmosphere_out_c,overwrite=True)

    # Alter settings file with updated TOA
    with open(settings_photochem_out,'r') as f:
        settings = yaml.load(f,Loader=Loader)
    settings['atmosphere-grid']['top'] = float(pc.var.top_atmos)
    settings = FormatSettings_main(settings)
    with open(settings_photochem_out,'w') as f:
        yaml.dump(settings, f, Dumper=MyDumper ,sort_keys=False, width=70)

    # write picaso file
    make_picaso_input_neptune(outfile)

    # make file for clouds
    haze_file = outfile+'_clouds.txt'
    make_cloud_file(pc, pc_q, P_trop, P_condense, haze_file)
    
def make_cloud_file(pc, pc_q, P_trop, P_condense, outfile):

    particle_radius = {}
    col = {}

    # water cloud
    particle_radius['H2O'] = 10 # microns
    log10P_cloud_thickness = np.log10(P_condense) - np.log10(P_trop) # water cloud thickness

    # Super rough based on Earth's troposphere
    # 100 particles/cm^3 over 10 km troposphere
    cloud_column_per_log10P = 100*10e5 # (particles/cm^2) / log10 pressure unit

    # total H2O cloud column in particles/cm^2
    tot_col_H2O_cloud = cloud_column_per_log10P*log10P_cloud_thickness

    trop_ind = np.argmin(np.abs(P_trop - pc.wrk.pressure))
    condense_ind = np.argmin(np.abs(P_condense - pc.wrk.pressure))

    # particles/cm^3
    density_H2O_cloud = tot_col_H2O_cloud/(pc.var.z[trop_ind] - pc.var.z[condense_ind])

    dz = pc.var.z[1] - pc.var.z[0]
    col_H2O = np.ones(pc.wrk.pressure.shape[0])*1e-100
    col_H2O[condense_ind:trop_ind] = density_H2O_cloud*dz

    assert np.isclose(np.sum(col_H2O),tot_col_H2O_cloud)

    col['H2O'] = col_H2O

    # HC cloud
    particle_radius['HC'] = 0.1 # microns
    ind = pc.dat.species_names.index('HCaer1')
    haze_density = pc.wrk.densities[ind,:]
    ind = pc.dat.species_names.index('HCaer2')
    haze_density += pc.wrk.densities[ind,:]
    ind = pc.dat.species_names.index('HCaer3')
    haze_density += pc.wrk.densities[ind,:]
    dz = pc.var.z[1] - pc.var.z[0]
    col_HC = haze_density*dz
    col['HC'] = col_HC

    # S2 and S8
    particle_radius['S'] = 0.1 # microns
    ind = pc.dat.species_names.index('S2aer')
    haze_density = pc.wrk.densities[ind,:]
    ind = pc.dat.species_names.index('S8aer')
    haze_density += pc.wrk.densities[ind,:]
    col_S = haze_density*dz
    col['S'] = col_S

    pressure = pc.wrk.pressure
    # Append values for lower atmosphere
    pressure = np.append(pc_q.wrk.pressure,pressure)
    for key in col:
        col[key] = np.append(np.ones(pc_q.wrk.pressure.shape[0])*1e-100,col[key])

    # remove last
    pressure = pressure[:-1].copy()
    for key in col:
        col[key] = col[key][:-1].copy()
    
    utils.make_haze_opacity_file(pressure, col, particle_radius, outfile)

def default_params():
    params = {}
    params['settings_quench_in'] = 'input/neptune/settings_neptune_quench_template.yaml'
    params['settings_photochem_in'] = 'input/neptune/settings_neptune_photochem_template.yaml'
    params['PTfile_in'] = 'results/neptune/climate/MH=2.000_CO=1.000_Tint=60.0.pkl'
    params['outfile'] = None
    params['P_bottom'] = 500.0e6
    params['P_top'] = 1.0e6
    params['M_H_metalicity'] = np.log10(100)
    params['CtoO'] = 1.0
    params['ct_file'] = 'input/zahnle_earth_new_ct.yaml'
    params['atoms'] = ['H','He','C','O','N']
    params['min_mix'] = 1.0e-8 # quench
    params['nz_q'] = 20
    params['eddy_q'] = 1.0e8 # from Hu (2021), page 6.
    params['T_trop'] = 215
    params['P_top_clima'] = 5.0e-4
    params['eddy_p'] = 5.0e5 # choosen arbitrarily
    params['equilibrium_time'] = 1e17
    return params

def nominal_S():
    params = default_params()
    params['outfile'] = 'results/neptune/nominal_S'
    params['atoms'] = ['H','He','C','O','N','S']
    params['equilibrium_time'] = 1.0e10
    params['eddy_p'] = 1e3
    return params

if __name__ == '__main__':
    run_quench_photochem_model(**nominal_S())
