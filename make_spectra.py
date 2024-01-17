import utils
import numpy as np
from picaso import justdoit as jdi
from scipy.stats import distributions
from scipy import optimize
from scipy.stats import norm
import os
import planets
import pickle
import yaml
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def compute_spectra(add_water_cloud, add_all_clouds, outfile):
    filename_db = os.path.join(os.getenv('picaso_refdata'), 'opacities','all_opacities_0.6_6_R60000.db')
    opa = jdi.opannection(filename_db=filename_db,wave_range=[0.01,100])
    opa = jdi.opannection(wave_range=[.01,100],filename_db=filename_db)
    case1 = jdi.inputs()
    case1.phase_angle(0)
    case1.gravity(mass=planets.k2_18b.mass, mass_unit=jdi.u.Unit('M_earth'),
                radius=planets.k2_18b.radius, radius_unit=jdi.u.Unit('R_earth'))
    case1.star(opa, planets.k2_18.Teff, planets.k2_18.metal, planets.k2_18.logg, radius=planets.k2_18.radius, 
            radius_unit = jdi.u.Unit('R_sun'),database='phoenix')
    case1.approx(p_reference=1.0)

    model_type = ['habitable','habitable','habitable','neptune','neptune']
    model_names = ['model1a','model1b','model1c','nominal_S','nominal_S_Kzz4']
    model_folders = [
        'results/habitable/',
        'results/habitable/',
        'results/habitable/',
        'results/neptune/',
        'results/neptune/'
    ]

    species_to_exclude = [['H2O'],['NH3'],['CO2'],['CH4'],['CO'],['HCN'],['C2H6'],['H2S']]
    res = {}
    for i in range(len(model_folders)):
        atmosphere_file = model_folders[i]+model_names[i]+'_picaso.pt'
        case1.atmosphere(filename = atmosphere_file, delim_whitespace=True)

        if add_water_cloud:
            # Get cloud region from settings file
            if model_type[i] == 'habitable':
                settings_file = model_folders[i]+model_names[i]+'_settings.yaml'
                settings = {'clouds':{'P-condense':1e6, 'P-trop': 28021.17671039341}}
            else:
                settings_file = model_folders[i]+model_names[i]+'_settings_photochem.yaml'
                with open(settings_file,'r') as f:
                    settings = yaml.load(f,Loader=yaml.Loader)

            # Add the cloud
            p_cloud_base = np.log10(settings['clouds']['P-condense']/1e6)
            p_coud_top = np.log10(settings['clouds']['P-trop']/1e6)
            cloud_thickness = p_cloud_base - p_coud_top
            case1.clouds(g0=[0.9], w0=[0.9], opd=[10], p=[p_cloud_base], dp=[cloud_thickness])

        if add_all_clouds:
            case1.clouds(filename=model_folders[i]+model_names[i]+'_clouds.txt', delim_whitespace=True)

        df = case1.spectrum(opa, full_output=True,calculation='transmission')
        wno_h, rprs2_h  = df['wavenumber'] , df['transit_depth']
        entry = {}
        entry['all'] = {}
        entry['all']['wv'] = 1e4/wno_h[::-1].copy()
        entry['all']['rprs2'] = rprs2_h[::-1].copy()
        for sp in species_to_exclude:
            case1.atmosphere(filename = atmosphere_file,exclude_mol=sp, delim_whitespace=True)
            df = case1.spectrum(opa, full_output=True,calculation='transmission')
            wno_h, rprs2_h  = df['wavenumber'] , df['transit_depth']
            key = '_'.join(sp)
            entry[key] = {}
            entry[key]['wv'] = 1e4/wno_h[::-1].copy()
            entry[key]['rprs2'] = rprs2_h[::-1].copy()

        res[model_names[i]] = entry

    with open(outfile,'wb') as f:
        pickle.dump(res,f)

def stats_objective(x, data_y, err, expected_y):
    return utils.chi_squared(data_y, err, expected_y+x[0])

def stats_objective_1(x, i, data, rprs2_soss, rprs2_g395h):
    tmp1 = utils.chi_squared(data['soss']['rprs2'][i:], data['soss']['rprs2_err'][i:], rprs2_soss[i:]+x[0])
    tmp2 = utils.chi_squared(data['g395h']['rprs2'][:], data['g395h']['rprs2_err'][:], rprs2_g395h[:]+x[1])
    return tmp1 + tmp2

def compute_statistics(infile, out_stats_file):

    i_values = [0,6]

    with open('data/osfstorage-archive/lowres.pkl','rb') as f:
        data = pickle.load(f)

    with open(infile,'rb') as f:
        models = pickle.load(f)

    models_binned = {}
    for i in i_values:
        # rebin models to data
        models_r = {}
        for model in models:
            models_r[model] = {}
            for case in models[model]:
                
                # regrid spectrum to data
                _, _, rprs2 = utils.rebin_picaso_to_data(models[model][case]['wv'], models[model][case]['rprs2'], data['all']['wv_bins'])

                # Save results
                models_r[model][case] = {}
                models_r[model][case]['wv'] = data['all']['wv']
                models_r[model][case]['rprs2'] = rprs2
                
                # Find offset
                init = np.array([1e-5])
                args = (data['all']['rprs2'][i:], data['all']['rprs2_err'][i:], rprs2[i:])
                sol = optimize.minimize(stats_objective, init, method = 'Nelder-Mead', args = args)
                assert sol.success
                models_r[model][case]['offset'] = sol.x[0]

                # stats
                dof = data['all']['wv'][i:].shape[0]
                chi2 = utils.chi_squared(data['all']['rprs2'][i:], data['all']['rprs2_err'][i:], rprs2[i:]+sol.x[0])
                rchi2 = chi2/dof
                p = distributions.chi2.sf(chi2, dof)
                sig = norm.ppf(1 - p)
                models_r[model][case]['rchi2'] = rchi2
                models_r[model][case]['p'] = p
                models_r[model][case]['sig'] = sig

                # Find split offset
                _, _, rprs2_soss = utils.rebin_picaso_to_data(models[model][case]['wv'], models[model][case]['rprs2'], data['soss']['wv_bins'])
                _, _, rprs2_g395h = utils.rebin_picaso_to_data(models[model][case]['wv'], models[model][case]['rprs2'], data['g395h']['wv_bins'])

                init = np.array([1e-5,1e-5])
                args = (i,data,rprs2_soss,rprs2_g395h)
                sol = optimize.minimize(stats_objective_1, init, method = 'Nelder-Mead', args = args)
                assert sol.success
                models_r[model][case]['split'] = {}
                models_r[model][case]['split']['wv_soss'] = data['soss']['wv']
                models_r[model][case]['split']['rprs2_soss'] = rprs2_soss
                models_r[model][case]['split']['wv_g395h'] = data['g395h']['wv']
                models_r[model][case]['split']['rprs2_g395h'] = rprs2_g395h
                models_r[model][case]['split']['offset_soss'] = sol.x[0]
                models_r[model][case]['split']['offset_g395h'] = sol.x[1]

                # stats
                dof = data['all']['wv'][i:].shape[0]
                chi2 = utils.chi_squared(data['soss']['rprs2'][i:], data['soss']['rprs2_err'][i:], rprs2_soss[i:]+sol.x[0]) \
                     + utils.chi_squared(data['g395h']['rprs2'][:], data['g395h']['rprs2_err'][:], rprs2_g395h[:]+sol.x[1])
                rchi2 = chi2/dof
                p = distributions.chi2.sf(chi2, dof)
                sig = norm.ppf(1 - p)
                models_r[model][case]['split']['rchi2'] = rchi2
                models_r[model][case]['split']['p'] = p
                models_r[model][case]['split']['sig'] = sig

        model = 'flat'
        case = 'all'
        models_r[model] = {}
        rprs2[:] = 0.002944
        models_r[model][case] = {}
        models_r[model][case]['wv'] = data['all']['wv']
        models_r[model][case]['rprs2'] = rprs2

        init = np.array([1e-5])
        args = (data['all']['rprs2'][i:], data['all']['rprs2_err'][i:], rprs2[i:])
        sol = optimize.minimize(stats_objective, init, method = 'Nelder-Mead', args = args)
        assert sol.success
        models_r[model][case]['offset'] = sol.x[0]

        dof = data['all']['wv'][i:].shape[0]
        chi2 = utils.chi_squared(data['all']['rprs2'][i:], data['all']['rprs2_err'][i:], rprs2[i:]+sol.x[0])
        rchi2 = chi2/dof
        p = distributions.chi2.sf(chi2, dof)
        sig = norm.ppf(1 - p)
        models_r[model][case]['rchi2'] = rchi2
        models_r[model][case]['p'] = p
        models_r[model][case]['sig'] = sig

        rprs2_soss[:] = 0.002944
        rprs2_g395h[:] = 0.002944
        init = np.array([1e-5,1e-5])
        args = (i,data,rprs2_soss,rprs2_g395h)
        sol = optimize.minimize(stats_objective_1, init, method = 'Nelder-Mead', args = args)
        assert sol.success
        models_r[model][case]['split'] = {}
        models_r[model][case]['split']['wv_soss'] = data['soss']['wv']
        models_r[model][case]['split']['rprs2_soss'] = rprs2_soss
        models_r[model][case]['split']['wv_g395h'] = data['g395h']['wv']
        models_r[model][case]['split']['rprs2_g395h'] = rprs2_g395h
        models_r[model][case]['split']['offset_soss'] = sol.x[0]
        models_r[model][case]['split']['offset_g395h'] = sol.x[1]

        # stats
        dof = data['all']['wv'][i:].shape[0]
        chi2 = utils.chi_squared(data['soss']['rprs2'][i:], data['soss']['rprs2_err'][i:], rprs2_soss[i:]+sol.x[0]) \
             + utils.chi_squared(data['g395h']['rprs2'][:], data['g395h']['rprs2_err'][:], rprs2_g395h[:]+sol.x[1])
        rchi2 = chi2/dof
        p = distributions.chi2.sf(chi2, dof)
        sig = norm.ppf(1 - p)
        models_r[model][case]['split']['rchi2'] = rchi2
        models_r[model][case]['split']['p'] = p
        models_r[model][case]['split']['sig'] = sig

        models_binned[i] = models_r

    with open(out_stats_file,'wb') as f:
        pickle.dump(models_binned,f)

if __name__ == '__main__':
    add_water_cloud = False
    add_all_clouds = False
    outfile = 'results/spectra/spectra.pkl'
    out_stats_file = 'results/spectra/spectra_stats.pkl'
    compute_spectra(add_water_cloud, add_all_clouds, outfile)
    compute_statistics(outfile, out_stats_file)

    add_water_cloud = False
    add_all_clouds = True
    outfile = 'results/spectra/spectra_cloudy.pkl'
    out_stats_file = 'results/spectra/spectra_cloudy_stats.pkl'
    compute_spectra(add_water_cloud, add_all_clouds, outfile)
    compute_statistics(outfile, out_stats_file)
    