import numpy as np
from matplotlib import pyplot as plt

from picaso import justdoit as jdi
from picaso import justplotit as jpi
from scipy.stats import distributions
from scipy.stats import norm
import picaso.opacity_factory as opa
import yaml
from photochem.clima import rebin
from photochemclima import PhotochemClima

import utils
import planets
import os
import pickle


def figure4():
    with open('data/osfstorage-archive/lowres.pkl','rb') as f:
        data = pickle.load(f)

    with open('results/spectra/spectra_stats.pkl','rb') as f:
        models_s = pickle.load(f)

    plt.rcParams.update({'font.size': 13})
    fig = plt.figure(constrained_layout=False,figsize=[10,5])
    fig.patch.set_facecolor("w")

    gs = fig.add_gridspec(100, 200)
    ax1 = fig.add_subplot(gs[:49, :99])
    ax2 = fig.add_subplot(gs[51:100, :99])
    ax3 = fig.add_subplot(gs[51:100, 101:])
    axs = [ax1,ax2,ax3]

    model_names = ['model1','model2','nominal_S']
    model_labels = ['Model 1\n(lifeless Hycean)', 'Model 2\n(inhabited Hycean)' ,'Model 3\n(mini-Neptune)']
    fig_letter = ['(a)','(b)','(c)']
    for jj,ax in enumerate(axs):

        i = 0
        
        elinewidth = 1
        capsize = 1.7
        capthick = 0.8
        ms = 1.5
        alpha=0.5
        
        xerr = data['soss']['wv_err']
        if jj == 0:
            label = 'NIRISS SOSS'
        else:
            label = ''
        ax.errorbar(data['soss']['wv'][i:], data['soss']['rprs2'][i:]*1e2, data['soss']['rprs2_err'][i:]*1e2, xerr[i:], ls='',alpha=alpha, 
                    marker='o', ms=ms,c='k',elinewidth=elinewidth,label=label,capsize=capsize,capthick=capthick)
        xerr = data['g395h']['wv_err']
        if jj == 0:
            label = 'NIRSpec G395H'
        else:
            label = ''
        ax.errorbar(data['g395h']['wv'], data['g395h']['rprs2']*1e2, data['g395h']['rprs2_err']*1e2, xerr, ls='',alpha=alpha, 
                    marker='o', ms=ms,c='grey',elinewidth=elinewidth,label=label,capsize=capsize,capthick=capthick)


        i = 0
        model = model_names[jj]
        case = 'all'
        split = models_s[i][model][case]['split']
        wv = split['wv_soss']
        offset = split['offset_soss']
        rprs2 = split['rprs2_soss']+offset
        wv_1 = split['wv_g395h']
        offset_1 = split['offset_g395h']
        rprs2_1 = split['rprs2_g395h']+offset_1
        wv = np.append(wv,wv_1)
        rprs2_all = np.append(rprs2,rprs2_1)
        rchi2 = split['rchi2']
        sig = split['sig']
        
        print('%.2f    %.2f      %.1f'%(rchi2, sig, (offset-offset_1)*1e6))

        rchi2_exclude = models_s[6][model][case]['split']['rchi2']
        sig_exclude = models_s[6][model][case]['split']['sig']
        
        note = model_labels[jj]
        ax.text(0.02, .97, note, \
            size = 10.5,ha='left', va='top',transform=ax.transAxes)
        note = '$\chi^{2}_{r} = $%.2f'%(rchi2)
        ax.text(0.98, .98, note, \
            size = 10.5,ha='right', va='top',transform=ax.transAxes)
        ax.plot(wv, (rprs2_all)*1e2, lw=1.5, c='k', ls='-')

        note = fig_letter[jj]
        ax.text(0.01, 0.01, note, \
            size = 18,ha='left', va='bottom',transform=ax.transAxes)
        
        species = ['H2O','CH4','CO2','NH3','CO']
        species_label = ['H$_2$O','CH$_4$','CO$_2$','NH$_3$','CO']
        colors = ['C0','C1','C2','C7','C4']
        for j,sp in enumerate(species):
            case = sp
            split = models_s[i][model][case]['split']
            rprs2 = split['rprs2_soss']+offset
            rprs2_1 = split['rprs2_g395h']+offset_1
            rprs2 = np.append(rprs2,rprs2_1)
            if jj == 1:
                label = species_label[j]
            else:
                label = ''
            ax.fill_between(wv, rprs2_all*1e2, (rprs2)*1e2, facecolor=colors[j], alpha=0.3, label=label)

    # legends
    ax = axs[0]
    ax.legend(ncol=1,bbox_to_anchor=(1.52,.45),loc='lower center')
    ax = axs[1]
    ax.legend(ncol=3,bbox_to_anchor=(1.52,1.05),loc='lower center')

    # labels
    ax = axs[0]
    ax.set_ylabel('Transit depth (%)')
    ax.set_xticklabels([])
    ax = axs[1]
    ax.set_xlabel('Wavelength (microns)')
    ax.set_ylabel('Transit depth (%)')
    ax = axs[2]
    ax.set_yticklabels([])
    ax.set_xlabel('Wavelength (microns)')

    for ax in axs:
        ax.set_ylim(.276,.314)
        ax.set_xlim(0.8,5.2)
        ax.grid(alpha=0.4)


    plt.subplots_adjust(wspace=0.03,hspace=0.05)

    plt.savefig('figures/figure4.pdf',bbox_inches='tight')

def main():
    figure4()

if __name__ == '__main__':
    main()