import numpy as np
from matplotlib import pyplot as plt
from photochem import Atmosphere
import neptune

def figure3():
    params = neptune.nominal_S()

    pc_eq = Atmosphere('input/zahnle_earth_new_noparticles.yaml',\
                        params['outfile']+'_settings_quench.yaml',\
                        "input/k2_18b_stellar_flux.txt",\
                        params['outfile']+'_atmosphere_quench.txt')

    pc1 = Atmosphere('input/zahnle_earth_new_noparticles.yaml',\
                        params['outfile']+'_settings_quench.yaml',\
                        "input/k2_18b_stellar_flux.txt",\
                        params['outfile']+'_atmosphere_quench_c.txt')

    pc2 = Atmosphere('input/zahnle_earth_new_S8.yaml',\
                    params['outfile']+'_settings_photochem.yaml',\
                    "input/k2_18b_stellar_flux.txt",\
                    params['outfile']+'_atmosphere_photochem_c.txt')
    
    plt.rcParams.update({'font.size': 15.5})
    fig,ax = plt.subplots(1,1,figsize=[7,5])
    fig.patch.set_facecolor("w")

    species = ['H2O','CH4','CO2','H2','CO','N2','NH3','HCN','H2S','SO2']
    names = ['H$_2$O','CH$_4$','CO$_2$','H$_2$','CO','N$_2$','NH$_3$','HCN','H$_2$S','SO$_2$']
    colors = ['C0','C1','C2','C3','C4','C5','C7','C6','C8','C9']
    for i,sp in enumerate(species):
        ind = pc1.dat.species_names.index(sp)
        tmp = pc1.wrk.densities[ind,:]/pc1.wrk.density
        ax.plot(tmp,pc1.wrk.pressure/1e6,label=names[i],c=colors[i], lw=2)

    for i,sp in enumerate(species):
        ind = pc_eq.dat.species_names.index(sp)
        tmp = pc_eq.wrk.densities[ind,:]/pc_eq.wrk.density
        ax.plot(tmp,pc_eq.wrk.pressure/1e6,ls=':',c=colors[i],alpha=0.7)

    for i,sp in enumerate(species):
        ind = pc2.dat.species_names.index(sp)
        tmp = pc2.wrk.densities[ind,:]/pc2.wrk.density
        ax.plot(tmp,pc2.wrk.pressure/1e6,ls='-',c=colors[i], lw=2)


    ax.axhline(pc1.wrk.pressure[-1]/1e6, c='k', ls='-', lw=3, alpha=0.5)

    ax1 = ax.twiny()
    ax1.plot(pc1.var.temperature, pc1.wrk.pressure/1e6,c='k', ls='--', lw=2)
    ax1.plot(pc2.var.temperature, pc2.wrk.pressure/1e6,c='k', ls='--', lw=2)

    ax1.set_xlabel('Temperature (K)')

    note = 'Model 3\nMetalicity = 100x solar\nC/O = solar'
    ax.text(.98, .98, note, \
        size = 15,ha='right', va='top',transform=ax.transAxes)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-8,1)
    ax.set_ylim(500,1e-7)
    ax.set_xticks(10.0**np.arange(-8,1,1))
    ax.set_yticks(10.0**np.arange(-7,3,1))
    ax.grid(alpha=0.4)
    ax.legend(ncol=1,bbox_to_anchor=(1,1.0),loc='upper left')
    ax.set_xlabel('Mixing Ratio')
    ax.set_ylabel('Pressure (bar)')

    plt.savefig('figures/figure3.pdf',bbox_inches='tight')

def main():
    figure3()

if __name__ == '__main__':
    main()
