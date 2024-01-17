import warnings
warnings.filterwarnings('ignore')
import picaso.justdoit as jdi
import astropy.units as u
from astropy import constants
import numpy as np
import planets
import utils
import pickle
from threadpoolctl import threadpool_limits
from pathos.multiprocessing import ProcessingPool as Pool

def make_outfile_name(mh, CtoO, tint):
    outfile = 'MH=%.3f_CO=%.3f_Tint=%.1f.pkl'%(mh, CtoO, tint)
    return outfile

class NeptuneClimate():

    def __init__(self):
        self.nlevel = 91
        self.nofczns = 1
        self.nstr_upper = 85
        self.rfacv = 0.5 
        self.p_bottom = 3 # log10(bars)
        self.database_dir = 'input/picaso/climate/'
        self.outfolder = 'results/neptune/climate/'

    def run_climate_model(self, mh, CtoO, tint):
        print(mh, CtoO, tint)

        # Get the opacity database
        if mh >= 0:   
            mh_str = ('+%.2f'%mh).replace('.','')
        else:
            mh_str = ('-%.2f'%mh).replace('.','')
        CtoO_str = ('%.2f'%CtoO).replace('.','')

        ck_db = self.database_dir+f'sonora_2020_feh{mh_str}_co_{CtoO_str}.data.196'
        opacity_ck = jdi.opannection(ck_db=ck_db)
        
        # Initialize climate run
        cl_run = jdi.inputs(calculation="planet", climate = True)

        # set gravity
        grav = utils.gravity(planets.k2_18b.radius*constants.R_earth.value*1e2, planets.k2_18b.mass*constants.M_earth.value*1e3, 0.0)/1e2
        cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
        
        # Set tint
        cl_run.effective_temp(tint) 

        # Set stellar properties
        T_star = planets.k2_18.Teff
        logg = planets.k2_18.logg #logg, cgs
        metal = planets.k2_18.metal # metallicity of star
        r_star = planets.k2_18.radius # solar radius
        semi_major = planets.k2_18b.a # star planet distance, AU
        cl_run.star(opacity_ck, temp =T_star,metal =metal, logg =logg, radius = r_star,
                    radius_unit=u.R_sun,semi_major= semi_major , semi_major_unit = u.AU, database='phoenix')

        # Initial temperature guess
        nlevel = self.nlevel # number of plane-parallel levels in your code
        Teq = planets.k2_18b.Teq # planet equilibrium temperature
        pt = cl_run.guillot_pt(Teq, nlevel=nlevel, T_int = tint, p_bottom=self.p_bottom, p_top=-6)
        temp_guess = pt['temperature'].values
        pressure = pt['pressure'].values

        nofczns = self.nofczns # number of convective zones initially. Let's not play with this for now.
        nstr_upper = self.nstr_upper # top most level of guessed convective zone
        nstr_deep = nlevel -2 # this is always the case. Dont change this
        nstr = np.array([0,nstr_upper,nstr_deep,0,0,0]) # initial guess of convective zones
        rfacv = self.rfacv

        # Set inputs
        cl_run.inputs_climate(temp_guess= temp_guess, pressure= pressure,
                        nstr = nstr, nofczns = nofczns , rfacv = rfacv)

        # Run model
        out = cl_run.climate(opacity_ck)

        # save output
        outfile = self.outfolder+make_outfile_name(mh, CtoO, tint)
        with open(outfile,'wb') as f:
            pickle.dump(out,f)

def main():
    threadpool_limits(limits=1)
    nc = NeptuneClimate()

    mhs = [2.0]
    CtoOs = [1.0]
    tints = [60.0]
    inputs = []
    for mh in mhs:
        for CtoO in CtoOs:
            for tint in tints:
                inputs.append((mh, CtoO, tint))
    
    def wrap(params):
        nc.run_climate_model(*params)
    p = Pool(1)
    p.map(wrap, inputs)

if __name__ == '__main__':
    main()
    





