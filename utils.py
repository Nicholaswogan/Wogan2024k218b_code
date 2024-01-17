import numpy as np
from scipy import constants as const
import miepython
from photochem.clima import rebin
import numba as nb
import pickle

@nb.cfunc(nb.double(nb.double, nb.double, nb.double))
def custom_binary_diffusion_fcn(mu_i, mubar, T):
    # Equation 6 in Gladstone et al. (1996)
    b = 3.64e-5*T**(1.75-1.0)*7.3439e21*np.sqrt(2.01594/mu_i)
    return b

def eddy_profile_like_Earth(log10P, log10P_trop):
    """Generates an eddy diffusion profile like Earth's.
    Pressures in log10 bars. Modeled off of Massie and Hunten (1981).
    """
    slope = (3.6 - 5.4)/(-1 - (-4))
    eddy_upper = 5.6
    eddy_trop = 5.0
    eddy_cold_trap = 3.6

    eddy = np.zeros(len(log10P))

    inds = np.where(log10P > log10P_trop)
    eddy[inds] = eddy_trop

    ind = np.max(inds) + 1

    x = log10P[ind]
    y = eddy_cold_trap
    b = y - slope*x

    for i in range(ind,len(log10P)):
        eddy[i] = slope*log10P[i] + b

    eddy[eddy>eddy_upper] = eddy_upper

    return 10.0**eddy

def eddy_profile_like_Jupiter(log10P):
    """Generates an eddy diffusion profile like Jupiter's.
    Based on Moses et al. (2005), Figure 15.
    Follows Gladstone et al. (1996).
    """

    slope = (4.6 - 8)/(-3 - (-10))
    
    eddy = np.zeros(log10P.shape[0])
    ind = np.argmin(np.abs(log10P - (-1)))

    intercept = 3.6 - slope*log10P[ind+1]

    eddy[:ind+1] = 3
    eddy[ind+1] = 3.6
    eddy[ind+2:] = log10P[ind+2:]*slope + intercept

    return 10.0**eddy

def simple_eddy_diffusion_profile(log10P, log10P_trop, Kzz_trop):
    """Simple eddy diffusion profile with a constant troposphere Kzz
    connected to a Kzz that increases above the tropopause as P^-0.5
    as roughly suggested by breaking gravity waves.
    """
    eddy = np.zeros(log10P.shape[0])

    ind = np.argmin(np.abs(log10P - log10P_trop))
    eddy[:ind+1] = np.log10(Kzz_trop)

    intercept = np.log10(Kzz_trop) + 0.5*log10P[ind]
    eddy[ind+1:] = intercept - 0.5*log10P[ind+1:]
    eddy = 10.0**eddy
    return eddy

def composition_from_metalicity(M_H_metalicity):

    # composition of the Sun (mol/mol)
    # From Table 8 in Lodders et al. (2009), "Abundances of the elements in the solar system"
    sun_mol = {
        'H': 0.921514888949834,
        'He': 0.07749066995740882,
        'O': 0.0004946606569284939,
        'C': 0.0002300986287941852,
        'N': 6.278165064228584e-05,
        'Si': 3.131024001891741e-05,
        'Mg': 3.101174053726646e-05,
        'Ne': 0.00010582900979606936,
        'Fe': 2.6994013922759826e-05,
        'S': 1.1755152117252983e-05
    }

    # Separate metals from non-metals
    metals = ['O','C','N','Si','Mg','Ne','Fe','S']
    non_metals = ['H','He']

    # Add up all metals
    mol_metals = 0.0
    for sp in metals:
        mol_metals += sun_mol[sp]

    # Compute the mol of H, metal and He of body
    mol_H_body = (10.0**M_H_metalicity * (mol_metals/sun_mol['H']) + 1.0 + sun_mol['He']/sun_mol['H'])**(-1.0)
    mol_metal_body = 10.0**M_H_metalicity * (mol_metals/sun_mol['H'])*mol_H_body
    mol_He_body = (sun_mol['He']/sun_mol['H'])*mol_H_body
    
    # Check everything worked out
    assert np.isclose((mol_metal_body/mol_H_body)/(mol_metals/sun_mol['H']), 10.0**M_H_metalicity)
    
    # Get metal composition
    metal_fractions = {}
    for sp in metals:
        metal_fractions[sp] = sun_mol[sp]/mol_metals

    # compute composition of the body
    mol_body = {}
    mol_body['H'] = mol_H_body
    mol_body['He'] = mol_He_body
    for sp in metals:
        mol_body[sp] = mol_metal_body*metal_fractions[sp]

    return mol_body

def composition_from_metalicity_for_atoms(atoms, M_H_metalicity):
    mol_body = composition_from_metalicity(M_H_metalicity)

    mol_tot = 0.0
    for atom in atoms:
        if atom in mol_body:
            mol_tot += mol_body[atom]

    mol_out = {}
    for atom in atoms:
        if atom in mol_body: 
            mol_out[atom] = mol_body[atom]/mol_tot
        else:
            mol_out[atom] = 0.0
    return mol_out

def equilibrium_temperature(stellar_radiation, bond_albedo):
    T_eq = ((stellar_radiation*(1.0 - bond_albedo))/(4.0*const.sigma))**(0.25)
    return T_eq 

def write_picaso_atmosphere(mix, outfile, species):
    fmt = '{:25}'
    with open(outfile,'w') as f:
        f.write(fmt.format('pressure'))
        f.write(fmt.format('temperature'))
        for sp in species:
            f.write(fmt.format(sp))
        f.write('\n')
    
        P = mix['press'][::-1]/1e6
        T = mix['temp'][::-1]
    
        for i in range(P.shape[0]):
            f.write(fmt.format('%e'%P[i]))
            f.write(fmt.format('%e'%T[i]))
            for j,sp in enumerate(species):
                f.write(fmt.format('%e'%(mix[sp][P.shape[0]-i-1])))
            f.write('\n')

def residuals(data_y, err, expected_y):
    return (data_y - expected_y)/err

def chi_squared(data_y, err, expected_y):
    R = residuals(data_y, err, expected_y)
    return np.sum(R**2)

def reduced_chi_squared(data_y, err, expected_y, dof):
    chi2 = chi_squared(data_y, err, expected_y)
    return chi2/dof

def rebin_picaso_to_data(wv, flux, wv_bins_data):
    "Rebins Picaso output to new wavelength bins."  
    d = np.diff(wv)
    wv_bins = np.array([wv[0]-d[0]/2] + list(wv[0:-1]+d/2.0) + [wv[-1]+d[-1]/2]).copy()
    flux_vals = flux.copy()

    assert wv_bins_data.shape[1] == 2

    flux_vals_new = np.empty(wv_bins_data.shape[0])
    for i in range(wv_bins_data.shape[0]):
        flux_vals_new[i] = rebin(wv_bins, flux_vals, wv_bins_data[i,:].copy())[0]
    return wv_bins, flux_vals, flux_vals_new

def make_haze_opacity_file_OLD(pressure, haze_column, outfile):

    particle_radius_um = 0.1 # microns
    particle_radius_cm = particle_radius_um*(1/1e6)*(1e2/1) # cm 
    
    fil = open('data/khare_tholins.dat')
    lines = fil.readlines()
    fil.close()
    
    wavelength = [] # microns
    m_real = []
    m_imag = []
    for line in lines[13:]:
        tmp = line.split()
        wavelength.append(float(tmp[1]))
        m_real.append(float(tmp[2]))
        m_imag.append(float(tmp[3]))
        
    wavelength = np.array(wavelength) # this is micro meter
    m_real = np.array(m_real)
    m_imag = np.array(m_imag)
    
    radius = particle_radius_um
    x = 2 * np.pi * radius / wavelength
    m = m_real - 1j*m_imag
    qext, qsca, qback, g = miepython.mie(m, x)
    w0 = qsca/qext
    
    keys = ['pressure','wavenumber','opd','w0','g0']
    out = {}
    for key in keys:
        out[key] = []

    for j in range(haze_column.shape[0]):
        for i in range(qext.shape[0]):
            taup_1 = qext[i]*particle_radius_cm**2*haze_column[j]
    
            out['pressure'].append(pressure[j]/1e6)
            out['wavenumber'].append(1e4/wavelength[i])
            out['opd'].append(taup_1)
            out['w0'].append(w0[i])
            out['g0'].append(g[i])

    fmt = '{:20}'
    with open(outfile,'w') as f:
        for key in out:
            f.write(fmt.format(key))
        f.write('\n')
        for i in range(len(out['pressure'])):
            for key in out:
                f.write(fmt.format('%e'%(out[key][i])))
            f.write('\n')

def make_haze_opacity_file(pressure, cols, particle_radius, outfile):
    # pressure in dynes/cm^2
    # cols in particles/cm^2
    # particle_radius in microns
    
    for key in ['S','HC','H2O']:
        assert key in cols
        assert key in particle_radius

    assert len(cols['S']) == len(pressure)
    assert len(cols['HC']) == len(pressure)
    assert len(cols['H2O']) == len(pressure)

    # Convert
    particle_radius_cm = {}
    for key in particle_radius:
        particle_radius_cm[key] = particle_radius[key]*(1/1e6)*(1e2/1) # convert from um to cm
    
    # Load all optical data
    with open('data/aerosol_optical_props.pkl','rb') as f:
        opt = pickle.load(f)
    
    # Compute optical properties with mie theory
    mie = {}
    for key in particle_radius:
        x = 2 * np.pi * particle_radius[key] / opt['wv']
        m = opt[key]['nr'] - 1j*opt[key]['ni']
        qext, qsca, qback, g = miepython.mie(m, x)
        w0 = qsca/qext
    
        mie[key] = {}
        mie[key]['qext'] = qext
        mie[key]['w0'] = w0
        mie[key]['g'] = g
    
    # At each altitude and wavelength, compute the optical
    # properties considering the particle density
    keys = ['pressure','wavenumber','opd','w0','g0']
    out = {}
    for key in keys:
        out[key] = []
    
    for j in range(pressure.shape[0]): # over altitude
        for i in range(opt['wv'].shape[0]): # over wavelength
            tausp_1 = {}
            taup = 0.0
            tausp = 0.0
            for k,key in enumerate(mie):
                taup_1 = mie[key]['qext'][i]*np.pi*particle_radius_cm[key]**2*cols[key][j]
                taup += taup_1
                tausp_1[key] = mie[key]['w0'][i]*taup_1
                tausp += tausp_1[key]
            gt = 0.0
            for k,key in enumerate(mie):
                gt += mie[key]['g'][i]*tausp_1[key]/(tausp)
            gt = np.minimum(gt,0.99999999)
            w0 = np.minimum(0.9999999,tausp/taup)
                
            out['pressure'].append(pressure[j]/1e6)
            out['wavenumber'].append(1e4/opt['wv'][i])
            out['opd'].append(taup)
            out['w0'].append(w0)
            out['g0'].append(gt)
            
    # Save the results
    fmt = '{:20}'
    with open(outfile,'w') as f:
        for key in out:
            f.write(fmt.format(key))
        f.write('\n')
        for i in range(len(out['pressure'])):
            for key in out:
                f.write(fmt.format('%e'%(out[key][i])))
            f.write('\n')

def haze_production_rate(pc):
    res = {}
    
    pl = pc.production_and_loss('HCaer1',pc.wrk.usol)
    ind = pl.production_rx.index('C2H + C4H2 => HCaer1 + H')
    res['HCaer1_prod'] = pl.integrated_production[ind]
    
    pl = pc.production_and_loss('HCaer2',pc.wrk.usol)
    ind = pl.production_rx.index('H2CN + HCN => HCaer2')
    res['HCaer2_prod'] = pl.integrated_production[ind]
    
    pl = pc.production_and_loss('HCaer3',pc.wrk.usol)
    ind = pl.production_rx.index('C4H + HCCCN => HCaer3')
    res['HCaer3_prod'] = pl.integrated_production[ind]

    # molecules/cm^2/s * mol/molecules * g/mol = g/cm^2/s
    haze_prod = 0.0
    ind = pc.dat.species_names.index('HCaer1')
    haze_prod += (res['HCaer1_prod']/const.Avogadro)*pc.dat.species_mass[ind] 
    ind = pc.dat.species_names.index('HCaer2')
    haze_prod += (res['HCaer2_prod']/const.Avogadro)*pc.dat.species_mass[ind] 
    ind = pc.dat.species_names.index('HCaer3')
    haze_prod += (res['HCaer3_prod']/const.Avogadro)*pc.dat.species_mass[ind] 
    res['haze_prod'] = haze_prod
    return res

def gravity(radius, mass, z):
    "CGS units"
    G_grav = const.G
    grav = G_grav * (mass/1.0e3) / ((radius + z)/1.0e2)**2.0
    grav = grav*1.0e2 # convert to cgs
    return grav # cm/s^2