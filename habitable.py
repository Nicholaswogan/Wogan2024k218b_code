import numpy as np
from threadpoolctl import threadpool_limits
from pathos.multiprocessing import ProcessingPool as Pool
from photochemclima import PhotochemClima
import pickle
import utils

def make_picaso_input_habitable(p, outfile):
    pc = p.pc
    mix = {}
    mix['press'] = pc.wrk.pressure
    mix['temp'] = pc.var.temperature
    for i,sp in enumerate(pc.dat.species_names[pc.dat.np:-2]):
        ind = pc.dat.species_names.index(sp)
        tmp = pc.wrk.densities[ind,:]/pc.wrk.density
        mix[sp] = tmp
    species = pc.dat.species_names[pc.dat.np:-2]
    utils.write_picaso_atmosphere(mix, outfile+'_picaso.pt', species)

def run_model(outfile, T_surf, mix, vdep, eddy, T_trop, relative_humidity,equilibrium_time,atol,atol_min,atol_max):

    p = PhotochemClima('input/zahnle_earth_new.yaml',
                   'input/habitable/settings_habitable_template.yaml',
                   'input/k2_18b_stellar_flux.txt',
                   'input/habitable/atmosphere_init.txt',
                   'input/habitable/species_climate.yaml',
                   'input/habitable/settings_climate_scale=0.7.yaml',
                   data_dir='/Users/nicholas/Documents/Research_local/PhotochemPy/photochem_clima_data')
    p.pc.var.verbose = 1
    p.pc.var.equilibrium_time=equilibrium_time
    p.pc.var.atol=atol
    p.atol_min = atol_min
    p.atol_max = atol_max

    # Other variables
    p.constant_eddy = eddy
    for sp in mix:
        if sp == 'H2O' or sp == 'H2':
            continue
        else:
            p.pc.set_lower_bc(sp,bc_type='mix',mix=mix[sp])
    for sp in vdep:
        p.pc.set_lower_bc(sp,bc_type='vdep',vdep=vdep[sp])
    p.relative_humidity = relative_humidity
    p.c.T_trop = T_trop

    # photochemical equilibrium
    res = p.find_equilibrium(T_surf, mix)

    # Write output file
    atmosphere_out_c = outfile+"_atmosphere.pkl"
    with open(atmosphere_out_c,'wb') as f:
        pickle.dump(res,f)
    
    # Write picaso file
    make_picaso_input_habitable(p, outfile)
    
    # haze column density (particles/cm^2)
    ind = p.pc.dat.species_names.index('HCaer1')
    haze_density = p.pc.wrk.densities[ind,:]
    ind = p.pc.dat.species_names.index('HCaer2')
    haze_density += p.pc.wrk.densities[ind,:]
    ind = p.pc.dat.species_names.index('HCaer3')
    haze_density += p.pc.wrk.densities[ind,:]
    dz = p.pc.var.z[1] - p.pc.var.z[0]
    haze_column = haze_density*dz
    pressure = p.pc.wrk.pressure
    haze_file = outfile+'_clouds.txt'
    make_cloud_file(p.pc, p.P_trop, haze_file)

def make_cloud_file(pc, P_trop, outfile):

    particle_radius = {}
    col = {}

    # water cloud
    particle_radius['H2O'] = 10 # microns
    log10P_cloud_thickness = np.log10(pc.var.surface_pressure*1e6) - np.log10(P_trop) # water cloud thickness

    # Super rough based on Earth's troposphere
    # 100 particles/cm^3 over 10 km troposphere
    cloud_column_per_log10P = 100*10e5 # (particles/cm^2) / log10 pressure unit

    # total H2O cloud column in particles/cm^2
    tot_col_H2O_cloud = cloud_column_per_log10P*log10P_cloud_thickness

    trop_ind = np.argmin(np.abs(P_trop - pc.wrk.pressure))

    # particles/cm^3
    density_H2O_cloud = tot_col_H2O_cloud/(pc.var.z[trop_ind] - pc.var.z[0])

    dz = pc.var.z[1] - pc.var.z[0]
    col_H2O = np.ones(pc.wrk.pressure.shape[0])*1e-100
    col_H2O[:trop_ind] = density_H2O_cloud*dz

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
    # ind = pc.dat.species_names.index('S2aer')
    # haze_density = pc.wrk.densities[ind,:]
    # ind = pc.dat.species_names.index('S8aer')
    # haze_density += pc.wrk.densities[ind,:]
    # col_S = haze_density*dz
    col['S'] = np.ones(pc.wrk.pressure.shape[0])*1e-100

    pressure = pc.wrk.pressure
    # remove last
    pressure = pressure[:-1].copy()
    for key in col:
        col[key] = col[key][:-1].copy()
    
    utils.make_haze_opacity_file(pressure, col, particle_radius, outfile)

def default_params():
    params = {}
    params['outfile'] = None
    params['T_surf'] = 320.0
    params['mix'] = {'H2O': 200.0, 'CO2': 0.008, 'N2': 1.0e-2}
    params['vdep'] = {'CO': 0.0}
    params['eddy'] = 5.0e5
    params['T_trop'] = 215.0
    params['relative_humidity'] = 1.0
    params['equilibrium_time'] = 1.0e17
    params['atol'] = 1.0e-27
    params['atol_min'] = 1.0e-29
    params['atol_max'] = 1.0e-26
    return params

def model1():
    params = default_params()
    params['outfile'] = 'results/habitable/model1'
    params['mix'] = {'H2O': 200.0, 'CO2': 0.008, 'N2': 3.0e-3}
    params['eddy'] = 5.0e5
    params['equilibrium_time'] = 1.0e14
    params['atol'] = 1.0e-24
    params['atol_min'] = 1.0e-25
    params['atol_max'] = 1.0e-24
    return params

def model2():
    params = default_params()
    params['outfile'] = 'results/habitable/model2'
    params['mix'] = {'H2O': 200.0, 'CO2': 0.008, 'N2': 3.0e-3, 'CH4': 2.0e-2}
    params['vdep'] = {'CO': 1.2e-4}
    params['eddy'] = 5.0e5
    return params

if __name__ == "__main__":
    threadpool_limits(limits=1)
    models = [
        model1,
        model2
    ]
    def wrap(model):
        run_model(**model())
    p = Pool(2)
    p.map(wrap, models)













