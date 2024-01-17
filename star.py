import numpy as np
from scipy import constants as const
from picaso import fluxes
import planets
from matplotlib import pyplot as plt
import utils
from astropy.io import fits

def stellar_radiation(wv, F):
    return 1e-3*np.sum(F[:-1]*(wv[1:]-wv[:-1]))

def main():
    # Stellar constant from Benneke et al. (2019)
    k218b_stellar_constant = planets.k2_18b.stellar_constant # W/m^2

    # GJ176 Spectrum
    spec = fits.getdata('star/hlsp_muscles_multi_multi_gj176_broadband_v22_adapt-const-res-sed.fits',1)
    wv = spec['WAVELENGTH']/10 # convert from Angstroms to nm
    F = spec['FLUX']*(1e10/1)*(1/1e9) # convert from erg/cm2/s/Ang to mW/m^2/nm
    
    # Add a blackbody spectrum to 100 microns
    wvb_cm = np.linspace((wv[-1]+wv[-1]*1e-5)/1e3,100,1000)/1e4
    wvb_nm = wvb_cm*(1/1e2)*(1e9/1) 
    Fb = fluxes.blackbody(planets.k2_18.Teff, wvb_cm)[0]*(1e2/1)*(1/1e9) # convert to mW/m^2/nm

    # Scale the blackbody, so it fits with the original spectrum
    factor = F[-1]/Fb[0]
    Fb = Fb*factor

    # Stitch together spectrum and blackbody
    wv_new = np.append(wv,wvb_nm)
    F_new = np.append(F,Fb)

    # Scale to match solar constant
    stellar_constant = stellar_radiation(wv_new, F_new)
    factor = k218b_stellar_constant/stellar_constant
    F_new = F_new*factor

    # Write the output file
    fmt = '{:30}'
    with open('input/k2_18b_stellar_flux.txt','w') as f:
        f.write(fmt.format('Wavelength (nm)'))
        f.write(fmt.format('Solar flux (mW/m^2/nm)'))
        f.write('\n')
        for i in range(F_new.shape[0]):
            f.write(fmt.format('%e'%wv_new[i]))
            f.write(fmt.format('%e'%F_new[i]))
            f.write('\n')

    # Print some useful information
    stellar_constant = stellar_radiation(wv_new, F_new)
    T_eq = utils.equilibrium_temperature(stellar_constant, 0.0)
    print('Bolometric flux at planet = %.1f W/m^2'%(stellar_constant))
    print('Equilibrium temperature = %.1f K'%(T_eq))

    # Plot
    plt.rcParams.update({'font.size': 15})
    fig,ax = plt.subplots(1,1,figsize=[6,5])
    fig.patch.set_facecolor("w")

    # Plot the spectrum
    ax.plot(wv_new,F_new,lw=.5, c='k', label='GJ176 spectrum scaled to K2-18b')

    # Plot blackbody spectrum at K2-18b
    wvb_cm = np.linspace(.1,100,10000)/1e4
    wvb_nm = wvb_cm*(1/1e2)*(1e9/1) 
    Fb = fluxes.blackbody(planets.k2_18.Teff, wvb_cm)[0]*(1e2/1)*(1/1e9) # convert to mW/m^2/nm
    stellar_constant_b = stellar_radiation(wvb_nm, Fb)
    factor = k218b_stellar_constant/stellar_constant_b
    Fb = Fb*factor
    ax.plot(wvb_nm,Fb,alpha=0.4,label='T = %i K blackbody'%(planets.k2_18.Teff))

    note = 'Bolometric flux = %.1f W/m$^2$\n'%(stellar_constant) \
          +'Equilibrium Temperature = %i K'%(T_eq)
    ax.text(.97, .02, note, \
            size = 11, ha='right', va='bottom',transform=ax.transAxes)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(1e1,1e5)
    ax.set_ylim(1e-5,2e3)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Stellar flux (mW m$^{-2}$ nm$^{-1}$)')
    ax.grid(alpha=0.8)
    ax.legend(ncol=1,bbox_to_anchor=(0.02,1.05),loc='lower left',fontsize=13)

    plt.savefig('figures/stellar_flux.pdf',bbox_inches='tight')

if __name__ == "__main__":
    main()



