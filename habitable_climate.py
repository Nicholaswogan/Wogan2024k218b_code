import numpy as np
from matplotlib import pyplot as plt
from photochem.clima import AdiabatClimate
from labellines import labelLine

def figure1():
    TT = np.linspace(216,640,100)
    P_i = np.array([1.0e6, 1.0e-10, 1.0e-10, 1.0, 1.0e-10, 1.0e-10])*1e6
    T_trop = 215
    res = {}

    c = AdiabatClimate('input/habitable/species_climate.yaml',
                  'input/habitable/settings_climate.yaml',
                  'input/k2_18b_stellar_flux.txt')

    c.T_trop = T_trop
    c.solve_for_T_trop = False
    c.RH = np.ones(len(c.species_names))*1
    albedos = [0.06]
    res['full'] = {}
    for albedo in albedos:
        c.rad.surface_albedo = albedo
        
        OLR = np.empty(TT.shape[0])
        ISR = np.empty(TT.shape[0])
        for i,T in enumerate(TT):
            ISR_, OLR_ = c.TOA_fluxes(T, P_i)
            OLR[i] = OLR_/1.0e3
            ISR[i] = ISR_/1.0e3
        
        res['full'][albedo] = {}
        res['full'][albedo]['T'] = TT
        res['full'][albedo]['OLR'] = OLR
        res['full'][albedo]['ISR'] = ISR

    total_solar_energy = 4*c.rad.wrk_sol.fdn_n[-1]/1e3 # W/m^2
    print('Solar energy relative to Modern = %.3f'%(total_solar_energy))

    c = AdiabatClimate('input/habitable/species_climate.yaml',
                    'input/habitable/settings_climate_scale=0.7.yaml',
                    'input/k2_18b_stellar_flux.txt')

    c.T_trop = T_trop
    c.solve_for_T_trop = False
    c.RH = np.ones(len(c.species_names))*1
    albedos = [0.06]
    res['part'] = {}
    for albedo in albedos:
        c.rad.surface_albedo = albedo
        
        OLR = np.empty(TT.shape[0])
        ISR = np.empty(TT.shape[0])
        for i,T in enumerate(TT):
            ISR_, OLR_ = c.TOA_fluxes(T, P_i)
            OLR[i] = OLR_/1.0e3
            ISR[i] = ISR_/1.0e3
        
        res['part'][albedo] = {}
        res['part'][albedo]['T'] = TT
        res['part'][albedo]['OLR'] = OLR
        res['part'][albedo]['ISR'] = ISR

    # plot
    plt.rcParams.update({'font.size': 15})
    fig,axs = plt.subplots(1,2,figsize=[11,4])

    ax = axs[0]
    fs = 12

    ax.plot(res['full'][0.06]['T'],res['full'][0.06]['OLR'],lw=2, c='k', label='Outgoing longwave')
    labelLine(ax.get_lines()[0],510,yoffset=+15,align=False,zorder=2.5,outline_color=None,fontsize=fs)

    ax.plot(res['full'][0.06]['T'],res['full'][0.06]['ISR'],lw=2, c='C3',ls='-', label='Incoming shortwave\n($I_0 = 1368$ W m$^{-2}$)')
    labelLine(ax.get_lines()[1],510,yoffset=+23,align=False,zorder=2.5,outline_color=None,fontsize=fs)
    ax.plot(res['part'][0.06]['T'],res['part'][0.06]['ISR'],lw=2, c='C3',ls='--', label='Incoming shortwave\n'+r'($I = 0.7 \times I_0 = 958$ W m$^{-2}$)')
    labelLine(ax.get_lines()[2],510,yoffset=+23,align=False,zorder=2.5,outline_color=None,fontsize=fs)

    ax.grid(alpha=0.4)
    ax.set_ylabel('Flux (W m$^{-2}$)')
    ax.set_xlabel('Surface Temperature (K)')
    ax.set_ylim(100,400)
    ax.set_xlim(min(TT),max(TT))

    note = 'Surface albedo = 0.06\nH$_2$ = 1 bar\nH$_2$O at saturation'
    ax.text(.97, .02,note, size = 13, ha='right', va='bottom',transform=ax.transAxes)
    ax.text(0.01, 0.98, '(a)', \
            size = 20,ha='left', va='top',transform=ax.transAxes)

    ax = axs[1]
    c.make_profile_bg_gas(320,P_i,1e6,'H2')
    ax.plot(c.T,c.P/1e6, c='k', lw=2)
    ax.set_yscale('log')
    ax.set_ylim(1,1e-8)
    ax.grid(alpha=0.4)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (bar)')
    ax.set_xlim(180,330)
    ax.text(0.01, 0.98, '(b)', \
            size = 20,ha='left', va='top',transform=ax.transAxes)

    plt.subplots_adjust(wspace=0.3)

    plt.savefig('figures/figure1.pdf',bbox_inches='tight')

def main():
    figure1()

if __name__ == '__main__':
    main()