from photochem import Atmosphere, PhotoException
from photochem.clima import AdiabatClimate
import numpy as np
import utils

class PhotochemClima():

    def __init__(self, species_file, settings_file, star_file, atmosphere_file,
                 clima_species_file, clima_settings_file, data_dir=None):
        
        self.pc = Atmosphere(species_file, settings_file, star_file, atmosphere_file, data_dir)
        self.pc.var.custom_binary_diffusion_fcn = utils.custom_binary_diffusion_fcn # set our special binary diffusion parameter
        self.pc.var.atol = 1.0e-27
        self.pc.var.verbose = False
        self.c = AdiabatClimate(clima_species_file, clima_settings_file, star_file, data_dir)
        self.c.solve_for_T_trop = False
        self.c.T_trop = 215.0 # default tropopause temp

        self.bg_gas = self.pc.dat.species_names[-3] # background gas name
        self.relative_humidity = 1.0 # default relative humidity
        self.constant_eddy = 1.0e5 # default eddy diffusion
        self.altitude_dependent_eddy = False
        # Tolerance for P-T profile
        self.T_tol = 1.0 # K
        self.edd_tol = 0.1 # 0.1 log unit
        self.max_dT = 0.0

        # Tolerance for TOA pressure
        self.max_TOA_p = 1.0e-8*1.0e6
        self.min_TOA_p = 1.0e-9*1.0e6
        self.avg_TOA_p = 5.0e-9*1.0e6

        # Integration settings
        self.nsteps_max = 100_000 # max number of steps before giving up
        self.nsteps_reinit = 5000 # number of steps before re-initializing integration
        self.nerrors_max = 10 # max number of integration errors before giving up
        self.atol_min = 1.0e-29 # 
        self.atol_max = 1.0e-26

        # Variables for later
        self.P = None
        self.T = None
        self.edd = None
        self.P_trop = None
        self.log10P_interp = None
        self.T_interp = None
        self.max_dT = 0.0

    def initialize_atmosphere(self, T_surf, mix):

        # Generate P-z-T-mix profile using clima
        f_i = np.ones(len(self.c.species_names))*1.0e-30
        for key in mix:
            ind = self.c.species_names.index(key)
            f_i[ind] = mix[key]
        P_surf = self.pc.var.surface_pressure*1.0e6
        P_i = f_i*P_surf
        self.c.RH = np.ones(len(self.c.species_names))*self.relative_humidity
        self.c.P_top = self.min_TOA_p
        self.c.make_profile_bg_gas(T_surf, P_i, P_surf, self.bg_gas)

        # Update photochem vertical grid to match clima
        self.pc.update_vertical_grid(TOA_alt=self.c.z[-1])
        
        # Interpolate clima mixing ratios to photochem P-T grid
        usol = np.ones(self.pc.wrk.usol.shape)*1.0e-40
        for i,sp in enumerate(self.c.species_names):
            if sp != self.bg_gas:
                ind = self.pc.dat.species_names.index(sp)
                usol[ind,:] = np.interp(self.pc.var.z,self.c.z,self.c.f_i[:,i])
        self.pc.wrk.usol = usol

        # Set P-T-edd profile
        self.P = np.append(self.c.P_surf + self.c.P_surf*1e-12, self.c.P)
        self.T = np.append(self.c.T_surf, self.c.T)
        if self.altitude_dependent_eddy:
            log10P = np.log10(self.P/1e6)
            log10P_trop = np.log10(self.c.P_trop/1e6)
            self.edd = utils.simple_eddy_diffusion_profile(log10P, log10P_trop, self.constant_eddy)
            self.edd[self.edd >= 1e6] = 1e6
        else:
            self.edd = np.ones(self.P.shape[0])*self.constant_eddy
        self.P_trop = self.c.P_trop
        self.log10P_interp = np.log10(self.P.copy()[::-1])
        self.T_interp = self.T.copy()[::-1]
        self.log10edd_interp = np.log10(self.edd.copy()[::-1])
        self.pc.set_press_temp_edd(self.P, self.T, self.edd, self.P_trop)

        # relative humidity
        self.pc.var.relative_humidity = self.relative_humidity

        # If necessary, update the vertical grid again
        if self.pc.wrk.pressure[-1] < self.min_TOA_p or self.pc.wrk.pressure[-1] > self.max_TOA_p:
            self.pc.update_vertical_grid(TOA_pressure=self.avg_TOA_p)

    def step(self):
        tn = self.pc.step()

        # Check if P-T profile is within tolerance
        T_p = np.interp(np.log10(self.pc.wrk.pressure.copy()[::-1]), self.log10P_interp, self.T_interp)
        T_p = T_p.copy()[::-1]
        self.max_dT = np.max(np.abs(T_p - self.pc.var.temperature))
        if self.max_dT > self.T_tol:
            # If not in tolerance, then re-set the the T-P profile
            self.pc.set_press_temp_edd(self.P, self.T, self.edd, self.P_trop)
            self.pc.initialize_stepper(self.pc.wrk.usol.copy())
            tn = 0.0

        # Check that P-edd profile is within tolerance
        log10edd_p = np.interp(np.log10(self.pc.wrk.pressure.copy()[::-1]), self.log10P_interp, self.log10edd_interp)
        log10edd_p = log10edd_p.copy()[::-1]
        self.max_dedd = np.max(np.abs(log10edd_p - np.log10(self.pc.var.edd)))
        if self.max_dedd > self.edd_tol:
            self.pc.set_press_temp_edd(self.P, self.T, self.edd, self.P_trop)
            self.pc.initialize_stepper(self.pc.wrk.usol.copy())
            tn = 0.0

        # Check if TOA pressure is within bounds
        if self.pc.wrk.pressure[-1] < self.min_TOA_p or self.pc.wrk.pressure[-1] > self.max_TOA_p:
            # If not, then regrid the atmosphere
            self.pc.update_vertical_grid(TOA_pressure=self.avg_TOA_p)
            self.pc.initialize_stepper(self.pc.wrk.usol.copy())
            tn = 0.0

        return tn

    def photochemical_equilibrium(self):
        
        self.pc.initialize_stepper(self.pc.wrk.usol)
        tn = 0.0
        nsteps_total = 0
        nsteps = 0
        nerrors = 0
        success = True
        while tn < self.pc.var.equilibrium_time:
            try:
                tn = self.step()
                nsteps += 1 # successful steps
                nsteps_total += 1
                if nsteps > self.nsteps_reinit:
                    # No good progress is being made. Lets reinitialize
                    self.pc.var.atol = 10.0**np.random.uniform(low=np.log10(self.atol_min),high=np.log10(self.atol_max))
                    self.pc.initialize_stepper(self.pc.wrk.usol)
                    nsteps = 0
                
            except PhotoException as e:
                # If there is an error, lets reinitialize where we are
                usol = np.clip(self.pc.wrk.usol,a_min=1.0e-40,a_max=np.inf)
                self.pc.initialize_stepper(usol)
                # Iterate error counter
                nerrors += 1

            if nerrors > self.nerrors_max:
                success = False
                break
            if nsteps_total > self.nsteps_max:
                success = False
                break

        self.pc.destroy_stepper()
        return success
    
    def equilibrium_result(self, success):
        out = {}
        out['top_atmos'] = self.pc.var.top_atmos
        out['trop_alt'] = self.pc.var.trop_alt
        out['temperature'] = self.pc.var.temperature.copy()
        out['z'] = self.pc.var.z.copy()
        out['pressure'] = self.pc.wrk.pressure.copy()
        out['edd'] = self.pc.var.edd.copy()
        out['usol'] = self.pc.wrk.usol.copy()

        if not success:
            for key in out:
                out[key] = np.nan

        return success, out
    
    def find_equilibrium(self, T_surf, mix):

        self.initialize_atmosphere(T_surf, mix)

        # Try with initial guess from clima
        success = self.photochemical_equilibrium()
        if success:
            return self.equilibrium_result(success)

        # Try with an empty atmosphere
        self.pc.wrk.usol = np.ones(self.pc.wrk.usol.shape)*1e-40
        success = self.photochemical_equilibrium()
        return self.equilibrium_result(success)





    
