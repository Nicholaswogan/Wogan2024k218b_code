import numpy as np
from matplotlib import pyplot as plt
import pickle

from photochemclima import PhotochemClima
import habitable

def make_PhotochemClima(params):
    p = PhotochemClima('input/zahnle_earth_new.yaml',
                       'input/habitable/settings_habitable_template.yaml',
                       'input/k2_18b_stellar_flux.txt',
                       'input/habitable/atmosphere_init.txt',
                       'input/habitable/species_climate.yaml',
                       'input/habitable/settings_climate_scale=0.7.yaml')
    p.pc.var.verbose = 1
    p.constant_eddy = params['eddy']
    p.relative_humidity = params['relative_humidity']
    p.pc.var.relative_humidity = params['relative_humidity']
    for sp in params['mix']:
        if sp == 'H2O' or sp == 'H2':
            continue
        else:
            p.pc.set_lower_bc(sp,bc_type='mix',mix=params['mix'][sp])
    for sp in params['flux']:
        p.pc.set_lower_bc(sp,bc_type='flux',flux=params['flux'][sp])
    for sp in params['vdep']:
        p.pc.set_lower_bc(sp,bc_type='vdep',vdep=params['vdep'][sp])

    p.c.RH = np.ones(len(p.c.species_names))*params['relative_humidity']
    p.c.T_trop = params['T_trop']

    p.initialize_atmosphere(params['T_surf'],params['mix'])

    with open(params['outfile']+'_atmosphere.pkl','rb') as f:
        res = pickle.load(f)
    success, out = res

    p.pc.update_vertical_grid(TOA_alt=out['top_atmos'])
    p.pc.set_temperature(out['temperature'], out['trop_alt'])
    p.pc.var.edd = out['edd']
    p.pc.wrk.usol = out['usol']
    p.pc.prep_atmosphere(p.pc.wrk.usol)
    return p

def figure2():
    params = habitable.model1()
    p1 = make_PhotochemClima(params)
    pc1 = p1.pc

    params = habitable.model2()
    p2 = make_PhotochemClima(params)
    pc2 = p2.pc

    pcs = [pc1,pc2]

    plt.rcParams.update({'font.size': 13})
    fig,axs = plt.subplots(1,2,figsize=[8,3],sharey=True)
    fig.patch.set_facecolor("w")

    species = ['H2O','CH4','CO2','H2','CO','N2','NH3','O']
    labels = ['H$_2$O','CH$_4$','CO$_2$','H$_2$','CO','N$_2$','NH$_3$','O']
    colors = ['C0','C1','C2','C3','C4','C5','C7','C6']
    ls = ['-','-','-','-','-','-','-','--']
    CH4_fluxes = ['0',r'$5 \times 10^{10}$']
    model_names = ['1','2']
    model_labels = ['(lifeless)', '(inhabited)']
    fig_letter = ['(a)','(b)','(c)']
    for i,ax in enumerate(axs):
        pc = pcs[i]
        for j,sp in enumerate(species):
            ind = pc.dat.species_names.index(sp)
            tmp = pc.wrk.densities[ind,:]/pc.wrk.density
            if i == 1:
                label = labels[j]
            else:
                label = None
            ax.plot(tmp,pc.wrk.pressure/1e6, c=colors[j],label=label, lw=2, ls=ls[j])

        note =  'surf. CH$_4$ flux = '+CH4_fluxes[i]
        ax.text(.98, .99, note, \
            size = 10,ha='right', va='top',transform=ax.transAxes)

        note = 'Model '+model_names[i]+'\n'+model_labels[i]
        ax.text(.98, .915, note, \
            size = 10,ha='right', va='top',transform=ax.transAxes)
        
        note = fig_letter[i]
        ax.text(0.15, 0.01, note, \
            size = 18,ha='left', va='bottom',transform=ax.transAxes)

    ax = axs[1]
    ax.legend(ncol=4,bbox_to_anchor=(-.1,1.0),loc='lower center',fontsize=11.5)

    ax = axs[0]
    ax.set_ylabel('Pressure (bar)')

    for ax in axs:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e-10,1.1e-0)
        ax.set_ylim(1e0,1e-8)
        ax.set_xticks(10.0**np.arange(-10,0,2))
        ax.set_yticks(10.0**np.arange(-8,1,2))
        ax.grid(alpha=0.4)
        ax.set_xlabel('Mixing Ratio')

    plt.subplots_adjust(wspace=0.07)

    plt.savefig('figures/figure2.pdf',bbox_inches='tight')

def main():
    figure2()

if __name__ == '__main__':
    main()

