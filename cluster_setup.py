import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from astropy.cosmology import Planck15 as p15
from astropy.cosmology import z_at_value
from astropy import units as u
from astropy import constants as c
import warnings
import pandas as pd

class ClusterSetup:
    """
        Generate a synthetic galaxy cluster with semi-realistic spatial and photometric properties.

        This class provides a physically-motivated model of a relaxed galaxy cluster, including:
        - Galaxy positions and distances following a radial density profile.
        - Galaxy stellar masses and effective radii based on scaling relations.
        - Galaxy morphological types (elliptical, lenticular, spiral) and Sersic indices.
        - Galaxy star formation histories (SFHs) incorporating ex-situ and in-situ quenching,
        as well as merger-driven starbursts.
        - Galaxy luminosities in multiple photometric bands via stellar population synthesis.
        - Peculiar velocities and cluster velocity dispersion consistent with virial equilibrium.

        Parameters
        ----------
        log_virial_mass_halo : float
            Logarithm (base 10) of the halo virial mass (M200) in solar masses.
        z : float
            Observation redshift of the cluster. Must be greater than 0 and less than or equal to 1.
        seed : int, optional
            Seed for the random number generator (default: 0).
        N : int or str, optional
            Number of galaxies that populate the cluster. By default, a number is approximated via
            a halo occupation distribution (HOD) model (default: "default").

        Attributes
        ----------
        virial_mass_halo : float
            Total virial mass of the halo in solar masses.
        z : float
            Redshift of the cluster.
        galaxy_number : int
            Number of galaxies in the cluster.
        distances : np.ndarray
            Radial distances of galaxies from the cluster center.
        positions : tuple of np.ndarray
            Cartesian positions (x, y, z) of galaxies.
        masses : np.ndarray
            Stellar masses of galaxies.
        activities : np.ndarray
            Boolean array indicating star-forming (True) or quiescent (False) galaxies.
        lenticular : np.ndarray
            Boolean array indicating lenticular galaxies.
        sersic_indices : list
            Sersic index for each galaxy.
        effective_radii : np.ndarray
            Effective radii of galaxies in kpc.
        ellipticities : np.ndarray
            Ellipticity of galaxies.
        angles : np.ndarray
            Position angles of galaxies in degrees.
        peculiar_velocities : np.ndarray
            Line-of-sight peculiar velocities of galaxies.
        velocity_dispersion : float
            Cluster velocity dispersion in km/s.
        """
    
    G_units = c.G.to(u.km**2 * u.Mpc * u.M_sun**(-1) * u.s**(-2)) #gravitational constant
    G = G_units.value
    H0 = p15.H0.value

    def __init__(
            self,
            log_virial_mass_halo: float,
            z: float,
            seed: int = 0,
            N: int | str = "default",
    ):
            self.virial_mass_halo = 10**log_virial_mass_halo
            self.z = z
            self.rng = np.random.default_rng(seed)

            if self.virial_mass_halo < 10**13.5:
                raise RuntimeError("Halo virial mass must be above 1e13.5 solar masses.")
            if self.z > 1:
                 warnings.warn("Simulation holds decently up to z ~ 1")

            self.M200 = self.virial_mass_halo

            self.H = p15.H(self.z).value
            self.R_vir_halo, self.rho_c = self.R_vir(self.z)

            #scale parameters used in the galaxy density profile from Qin et al.
            self.a = 0.0778*np.log10(self.virial_mass_halo)-0.4419
            self.b = 3331.4392*np.exp(-0.5123*np.log10(self.virial_mass_halo))
            self.c = 3.51 * ((1+z)/1.35)**(-0.35) * (self.virial_mass_halo/1e15)**(-0.08) #this is a relation from darragh-ford et al. 2023 using cluster samples from cosmological n point simulations in child et al. 2018
            #valid for halos with M200 > 2*10^11 Msol and 0<= z <= 4

            self.M500 = self.get_M500()

            #load instances of most quantities needed for further setup into the class
            if isinstance(N, str) and N == "default":
                self.galaxy_number = int(self.get_galaxy_number())
            else:
                 self.galaxy_number = N
            self.distances = self.galaxy_distances()
            self.positions = self.galaxy_positions()
            self.masses = self.galaxy_masses()
            self.activities = self.galaxy_activity()
            self.activities = np.array(self.activities)
            self.peculiar_velocities, self.velocity_dispersion = self.galaxy_velocities(self.z)
            self.lenticular = np.array(self.lenticulars())
            self.sersic_indices = self.galaxy_sersic_indices()
            self.effective_radii = np.array([self.galaxy_effective_radii(m, self.z, "early") if not a else self.galaxy_effective_radii(m, self.z, "late") 
                                             for m, a in zip(self.masses, self.activities)])
            self.ellipticities = self.galaxy_ellipticities()
            self.angles = self.galaxy_angles()


    def R_vir(
              self,
              z
    ):
         
            rho_c = 3*p15.H(z).value**2/(8*np.pi*self.G) #critical density of the universe
            R_vir = (3*self.virial_mass_halo/(4*np.pi*200*rho_c))**(1/3) #virial radius with overdensity parameter = 200
            return R_vir, rho_c


    def get_M500(
            self
    ):
            rs = self.R_vir_halo/self.c
            f = lambda x : x**3 * (np.log(1 + 1/x) - 1 / (1+x))
            rho_s = self.M200/(4 * np.pi * self.R_vir_halo**3 * f(rs/self.R_vir_halo))
            diff = lambda r: 4 * np.pi * rho_s * r**3 * f(rs/r) - (4/3) * 500 * self.rho_c * r**3
            lo, hi=1e-8, self.R_vir_halo

            for _ in range(60):
                mid = 0.5 * (lo+hi)
                if diff(lo) * diff(mid) <=0: 
                    hi = mid
                else: 
                    lo = mid

            R500 = mid
            M500 = 4 * np.pi * rho_s * R500**3 * f(rs/R500)
            return M500

    def get_galaxy_number(
              self
    ):
        
            #from zheng et al. 2007 with generic best fit parameters, picked for the lowest magnitude bins
            if self.z >= 0.5:
                M_min = 11.83
                sig = 0.3
                M_0 = 10**11.53
                M_1_prime = 10**13.02
                alpha = 0.97

            elif self.z<0.5:
                M_min = 11.75
                sig = 0.28
                M_0 = 10**11.69
                M_1_prime = 10**13.01
                alpha = 1.06

            N_central = 1/2 * (1+erf((np.log10(self.virial_mass_halo)-M_min)/sig))
            N_sat = 1/2 * (1+erf((np.log10(self.virial_mass_halo)-M_min)/sig)) * ((self.virial_mass_halo-M_0)/M_1_prime)**alpha

            N = N_central + N_sat
            
            #in order to guarantee the 5NN algorithm in morphology works
            if N >= 6:
                return N
            else:
                return 6


    def galaxy_distances(
              self
    ):

        def galaxy_distribution(
                distance
            ):
                gal_density = distance**2 * np.exp(-self.b*(self.c*distance)**self.a)
                return distance ** 2 * gal_density #both density as well as this expression for the unnormalized distribution given by Qin et al. 2023

        r_max = 10*self.R_vir_halo #10 R_vir was chosen in the paper to ensure that the entire curve is integrated for the normalization factor
        norm_factor,_ = quad(galaxy_distribution,0,r_max)

        def galaxy_PDF(
                distance
            ):
                return galaxy_distribution(distance)/norm_factor

        #variables used in rejection sampling
        r_grid = np.linspace(0,r_max,10000)

        PDF_max = np.max(galaxy_PDF(r_grid))
        samples = [0]
        num_accepted = 0
            
        while num_accepted < self.galaxy_number-1:
            proposed = self.rng.uniform(0,r_max)
            limit = self.rng.uniform(0,PDF_max)
            if limit < galaxy_PDF(proposed):
                samples.append(proposed)
                num_accepted += 1

        return np.sort(np.array(samples))
    


    def galaxy_positions(
                self
    ):
        phi = self.rng.uniform(0, 2 * np.pi, self.galaxy_number) #distribution is radially symmetrical, so an angle can be uniformly generated to obtain the cartesian positions
        cos_theta = self.rng.uniform(-1,1, self.galaxy_number) #since the volume element scales with sin(theta), uniform generation would be biased towards poles -> direct generation of cosines
        theta = np.arccos(cos_theta)
        x = self.distances * np.cos(phi) * np.sin(theta)
        y = self.distances * np.sin(phi) * np.sin(theta)
        z = self.distances * cos_theta

        return x,y,z


    def galaxy_masses(
                self
    ):

        h = self.H0/100

        #segregation into different groups from Contini, Kang
        if 1e13 <= self.M200/h <= 5*1e13:
            a, b = self.rng.normal(-0.678, 0.012, self.galaxy_number), self.rng.normal(10.797, 0.005, self.galaxy_number)
        elif 5*1e13 < self.M200/h <= 1e14:
            a, b = self.rng.normal(-0.483, 0.023, self.galaxy_number), self.rng.normal(10.728, 0.011, self.galaxy_number)
        elif 1e14 < self.M200/h <= 5*1e14:
            a, b = self.rng.normal(-0.339, 0.021, self.galaxy_number), self.rng.normal(10.665, 0.011, self.galaxy_number)
        elif self.M200/h > 5*1e14:
            a, b = self.rng.normal(-0.227, 0.045, self.galaxy_number), self.rng.normal(10.681, 0.023, self.galaxy_number)        

        ms_sat = 10**np.array(a * (self.distances/self.R_vir_halo) + b)

        #slope and normalization parameters from Kravtsov et al.
        slope_bcg, norm_bcg = self.rng.normal(0.33, 0.11), self.rng.normal(12.24, 0.04)

        #linear log relation from Kravtsov et al.
        log_m_bcg = slope_bcg * (np.log10(self.M500)-14.5) + norm_bcg
        scatter_bcg = self.rng.normal(0.17, 0.03)
        scatter_bcg = max(scatter_bcg, 1e-4)  

        gal_masses = ms_sat
        gal_masses[0] = 10**self.rng.normal(float(log_m_bcg), float(scatter_bcg))

        return np.array(gal_masses)


    def galaxy_activity(
              self
    ):
        #the approach in Peng et al. (2010) requires a 5th nearest neighbour algorithm which applies to galaxies within cylinders of 1000km/s in
        #redshift space, however the latter should be redundant in an approximating simulation as the comoving coordinates are precisely determined.
        #the 5NN algorithm is implemented here, based on Peng et al. (2010) and Kovac et al. (2010), but modified for a volumetric profile
        #instead of a projected one, since membership is inherently given no tracer galaxies need to be chosen
        d = self.distances[:, np.newaxis]
        diffs = np.abs(d-d.T)
        diffs_sorted = np.sort(diffs, axis=1)
        r5 = diffs_sorted[:, 5]

        rho = 5/((4/3) * np.pi * r5**3) #local galaxy density

        #the global density is evaluated using a sphere with distance to the furthest galaxy as radius here 
        N = self.galaxy_number
        rho_m = N / ((4 / 3) * np.pi * self.distances[-1]**3)

        overdensity = np.clip((rho-rho_m)/rho_m, 0, None)

        #parameters for the fitting function
        z_centers = np.array([0.1/2, (0.35+0.1)/2, (0.5+0.35)/2, (0.5+0.7)/2, (1+0.7)/2])
        p1_vals = np.array([1.84, 1.84, 1.86, 1.74, 1.9])
        p2_vals = np.array([0.6, 0.6, 0.67, 0.7, 0.64])
        p3_vals = np.array([10.56, 10.78, 10.76, 10.83, 10.89])
        p4_vals = np.array([0.8, 0.78, 0.61, 0.65, 0.63])

        p_1 = 10**np.interp(self.z, z_centers, p1_vals)
        p_2 = np.interp(self.z, z_centers, p2_vals)
        p_3 = 10**np.interp(self.z, z_centers, p3_vals)
        p_4 = np.interp(self.z, z_centers, p4_vals)

        f_red = 1-np.exp(-(overdensity/p_1)**p_2-(self.masses/p_3)**p_4)
        active = [False if f > self.rng.random() else True for f in f_red]

        return active


    def lenticulars(
              self
    ):

        virial_mass_halo_log = np.log10(self.virial_mass_halo)

        lower_mass_bound = 13.5
        upper_mass_bound = np.log10(2.1*10**15)
        grid_cluster_masses = np.linspace(lower_mass_bound, upper_mass_bound, 6)
        grid_cluster_zs = np.linspace(0, 1, 6)

        red_fracs = np.array([
             [44.24, 35.43, 35.27, 30.83, 28.26, 26.93],
             [59.19, 49.55, 47.89, 41.88, 38.29, 36.21],
             [77.09, 69.84, 67.69, 58.42, 52.07, 49.20],
             [90.98, 87.70, 86.77, 80.82, 75.81, 72.86],
             [95.99, 94.47, 94.10, 91.20, 88.60, 86.95],
             [98.11, 97.44, 97.18, 95.76, 94.60, 93.80]
        ]) / 100

        #for now the reference value will just be the low end of the lowest mass bin at z=1 minus a baseline lenticular fraction
        f_elliptical_ref = red_fracs[0,5] - self.rng.uniform(15, 20)

        #builds a scipy grid-interpolator for the masses/z grid, using linear interpolation
        grid_interp = RegularGridInterpolator((grid_cluster_zs, grid_cluster_masses), 
                                              red_fracs.T, "linear", bounds_error=False, fill_value=None)
        
        lenticular_fraction = grid_interp((self.z, virial_mass_halo_log)) - f_elliptical_ref

        lenticular_bools = [self.rng.uniform(0,1) <= lenticular_fraction if not a else False for a in self.activities]
        lenticular_bools[0] = False

        return lenticular_bools

    #notes: maybe update this to have some mass dependency but this could be fine like that too.
    def galaxy_sersic_indices(
              self
    ):
        sersic_idc = []
        for a, l in zip(self.activities, self.lenticular):
            if a:
                sersic_idc.append(1)
            elif l:
                sersic_idc.append(self.rng.uniform(2,3))
            else:
                sersic_idc.append(4)
        
        return sersic_idc


    def time_bins_SFH(
              self
    ):

        t_univ = p15.age(self.z).value #cosmic epoch at z
        z_start = 3
        t_start = p15.age(z_start).value

        #setting up the age-grid
        n_time_bins = 1000
        age_bins = np.linspace(0, t_univ-t_start, n_time_bins+1) #in Gyr
        age_centers = 0.5*(age_bins[:-1] + age_bins[1:]) + t_start


        # --------------------- Setting up all the parameters for a delayed - then quenched model as in Rhee et al. ------------------ 

        # -------------------------------------------------------- delay time setup -------------------------------------------------

        #delayed quenching phase where galaxy not yet impacted by RPS/here merger
        #Rhee et al. argue their numbers are (at z~0.08) consistent with crossing time from R = 1.5Rvir
        def t_delay(
                  z
        ):
             _, v_disp = self.galaxy_velocities(z)
             velocity_dispersion_mpc_gyr = (v_disp * u.km / u.s).to(u.Mpc / u.Gyr).value
             return 1.5*self.R_vir(z)[0]/velocity_dispersion_mpc_gyr

        # ------------------------------------------------- Infall times, delay times  --------------------------------------------

        #long tails gamma distribution
        def p_infall(
                  t,
                  m
        ):
            tau = p15.H0.value/self.H * 1.75 * (m/1e10)**-0.07
            return ((t_univ-t)**2) * np.exp(-(t_univ-t)/tau)

        #normalization to pdf
        def infall_pdf(
                  m,
        ):
            N, _ = quad(lambda t: p_infall(t, m), t_start, t_univ)
            return lambda t: p_infall(t, m) / N

        def cdf_sampler_infall(
                  m        
        ):
            t_grid_infall = np.linspace(t_start, t_univ, 1000)
            f_pdf = infall_pdf(m)
            pdf_vals_infall = f_pdf(t_grid_infall)
            cdf = np.cumsum(pdf_vals_infall) * (t_grid_infall[1]-t_grid_infall[0])
            cdf /= cdf[-1]

            return np.interp(self.rng.random(), cdf, t_grid_infall)

        #rhee et al. model defines infall as first arrival at 1.5R_vir, which is used here as a boundary for the start of the delay time -> crossing time into rapid quenching regions
        t_infall = []
        ts_delay = []
        for a, r, m in zip(self.activities, self.distances, self.masses):
            if a and r>=1.5*self.R_vir_halo:
                t = t_univ
                t_infall.append(t)
                ts_delay.append(None)

            elif a and r<1.5*self.R_vir_halo:
                t = t_univ - t_delay(self.z) * (1.5*self.R_vir_halo - r) / (1.5*self.R_vir_halo) #linear approximation of active galaxy infall times by position in the cluster, results in TSI < t_delay -> no rapid quenching yet
                t_infall.append(t)
                ts_delay.append(t_delay(self.z))

            elif not a:
                t = cdf_sampler_infall(m)
                t_infall.append(t)
                ts_delay.append(t_delay(z_at_value(p15.age, t*u.Gyr)))

        t_since_infall = [t_univ-t for t in t_infall]

        #e-folding timescale for ex-situ effects such as (AGN-) feedback processes and group preprocessing
        #this might be imprecise as the free parameter alpha is scaled for an initial time at z=2, whereas I am starting at z=3
        #to include the expected fainter tail of the BCG 
        #maybe it is better to blindly extrapolate this alpha to z=3, since we also have the z=1 measurement, even though we don't know how exactly it evolves
        alpha = -0.66
        tau_ex_situ = [3.3*(m/1e10)**-alpha for m in self.masses] #reference point 3.3Gyr is the approximate tau calculated for 10^10 solar mass galaxies

        #starting points for the ex situ quenching phases of each galaxy determined with infall times and standard ex-situ durations
        t_quench_ex_situ = self.rng.normal(2.7, 1.1, self.galaxy_number)
        start_ex_situ = np.array([t_univ - tqxs- tsi for tqxs, tsi in zip(t_quench_ex_situ, t_since_infall)])
        start_ex_situ[start_ex_situ < t_start] = t_start

        #e-folding timescale for RPS inside the cluster, maybe in the future a mass scaling as M**-1 could be implemented here
        tau_cluster = [self.rng.uniform(0.7, 1.5) for _ in range(self.galaxy_number)]


        # -------------------------- Setup for mergers ----------------------------


        #ongoing merger fraction as function of redshift as in Ryan et al. 2007
        #robust within 0.5<z<2.5, holds with observation for z<0.5, very few mergers generally happen at z > 2.5 in a surviving pop at z in [0,1]
        def merger_fraction(
                z
         ):
            return 0.44*(1+z)**7 * np.exp(-3.1*(1+z))
         
        norm_factor,_ = quad(merger_fraction, 0, 3)
        
        def merger_pdf(
                z
        ):
            return merger_fraction(z)/norm_factor

        #building the cdf for inverse transform sampling
        z_grid_mergers = np.linspace(0, 3, 1000)
        pdf_vals_mergers = merger_pdf(z_grid_mergers)
        cdf = np.cumsum(pdf_vals_mergers) * (z_grid_mergers[1]-z_grid_mergers[0])
        cdf /= cdf[-1]

        #generate merger redshifts for elliptical galaxies by inverse transform sampling approximation and convert to age
        merger_times=[]
        merger_z=[]
        for idx, (a, l) in enumerate(zip(self.activities, self.lenticular)):
            if not a and not l:
                z_merger = np.interp(self.rng.random(), cdf, z_grid_mergers)
                t_merger = p15.age(z_merger).value

                if t_merger < (start_ex_situ[idx]):
                     t_merger = start_ex_situ[idx]
                     z_merger = z_at_value(p15.age, t_merger * u.Gyr)

                elif t_merger > (start_ex_situ[idx] + t_quench_ex_situ[idx]):
                     t_merger = start_ex_situ[idx] + t_quench_ex_situ[idx]
                     z_merger = z_at_value(p15.age, t_merger * u.Gyr)
                merger_z.append(z_merger)
                merger_times.append(p15.age(z_merger).value)
                
            else:
                 merger_times.append(None)
                 merger_z.append(None)

        #generate burst amplitudes using a simple cdf sampling approach for a probability distribution ~ 1/t from 2 to 5
        amp_burst = np.array([2*5**self.rng.random() if (not a and not l) else None for a, l in zip(self.activities, self.lenticular)])

        #using the paper Hopkins, Hernquist (2010), a lowest order estimation for amplification half life is given as ~10^8 yr with scatter 2-3
        #refining the simulation on a higher level could be tying the burst amplification factor to stellar mass as in eq. 9, calculating half life with the given fit
        t_half_generic = 0.11
        sigma_t_half_log10 = 0.3
        draws_log10 = self.rng.normal(np.log10(t_half_generic), sigma_t_half_log10, self.galaxy_number)
        tau_burst_amp = (10**draws_log10) / np.log(2)
        
        #quenching timescale for post-merger quenching
        tau_burst_quench = np.array([self.rng.uniform(0.96, 1.76) if z is not None else None for z in merger_z])


        # ---------------- SFR-setup functions -----------------


        #SFH from Speagle et al. 2014, valid up to z=6 in principle
        def SFR_init(
                t,
                m
            ):
                
                val1 = self.rng.normal(0.84, 0.02)
                val2 = self.rng.normal(0.026, 0.003)
                val3 = self.rng.normal(6.51, 0.24)
                val4 = self.rng.normal(0.11, 0.03)

                log_SFR = (val1-val2*t) * np.log10(m) - (val3-val4*t)
                return 10**log_SFR
         
        #ex-situ quenched mode for SFMS-SFR, includes effects such as preprocessing in group environments and self-quenching (AGN)
        #applied here to all galaxies but especially as exclusive quenching phase for spirals/blue galaxies
        def SFR_ex_situ(
                t,
                start_ex,
                m,
                tau_ex
         ):
              base_sfr = SFR_init(t, m)

              if t < start_ex:
                   return base_sfr
              else:
                    return base_sfr * np.exp(-(t-start_ex)/tau_ex)
         
        #in-situ quenched mode for SFMS-SFR, models rapid quenching through ram pressure stripping
        #applied here to lenticular galaxies
        def SFR_in_situ(t,
            start_ex,
            m,
            tsi,
            tau_ex,
            t_d,
            tau_c
        ):
            base_sfr = SFR_ex_situ(t, start_ex, m, tau_ex)
            t_quench_start = t_univ - tsi + t_d 
            
            if t < t_quench_start:
                return base_sfr
            else:
                return base_sfr * np.exp(-(t-t_quench_start)/tau_c)
         
        #in-situ quenched mode combined with a prior rapidly decaying exponential term amplifying SFR modelling starburst
        #applied here to ellipticals
        def SFR_starburst(
                t,
                t_burst, 
                tau_sigma, 
                tau_merger_quench, 
                amp_factor,
                start_ex, 
                m, 
                tsi, 
                tau_ex, 
                t_d, 
                tau_c
        ):
            base = SFR_in_situ(t, start_ex, m, tsi, tau_ex, t_d, tau_c)
            amp = 1 + amp_factor * np.exp(-0.5 * ((t-t_burst)/tau_sigma)**2)

            if t > t_burst + 3*tau_sigma:
                drop = np.exp(-(t-(t_burst + 3*tau_sigma))/tau_merger_quench)
                amp =  1 + amp_factor * np.exp(-0.5 * ((t-t_burst)/tau_sigma)**2) * drop
            return base * amp
         
        #loss fraction for Chabrier IMF according to Behroozi et al. 2019
        def loss_fraction(
                t
        ):
            
            return 0.05 * np.log(1 + (t_univ - t) * 1000/1.4)

        SFH = []


        # ----------- computing the initial mass and SFH ------------

        dt_gyr = np.diff(age_centers)[0] 
        dt_years = dt_gyr * 1e9

        for idx, (tsi, tau_ex, tau_c, lenticular, m_stel, start_ex, t_d, t_merger,
                A_burst, tau_merger_amp, tau_merger_quench) in enumerate(
                    zip(t_since_infall, tau_ex_situ, tau_cluster,
                        self.lenticular, self.masses, start_ex_situ, ts_delay, merger_times,
                        amp_burst, tau_burst_amp, tau_burst_quench)):

            target_final_mass = float(m_stel)

            #forward integrate mass from initial m0 to final mass
            def final_mass_from_m0(m0):
                mcur = float(m0)

                for t in age_centers:
                    #pick appropriate SFR branch (SFR in M_sun / yr)
                    if t_d is None or tsi <= t_d:  #ex-situ/blue branch
                        sfr = SFR_ex_situ(t, start_ex, mcur, tau_ex)
                    elif (tsi > t_d) and lenticular:   #in-situ/lenticular branch
                        sfr = SFR_in_situ(t, start_ex, mcur, tsi, tau_ex, t_d, tau_c)
                    elif (tsi > t_d) and not lenticular:  #starburst/elliptical branch
                        sfr = SFR_starburst(t, t_merger, tau_merger_amp, tau_merger_quench, A_burst,
                                            start_ex, mcur, tsi, tau_ex, t_d, tau_c)

                    #mass added this time bin: SFR [M_sun/yr] * dt [yr] * (1 - loss_fraction)
                    loss = loss_fraction(t)   # loss_fraction expects Gyr
                    delta_m = sfr * dt_years * (1.0 - loss)

                    #update mass, guard against NaN/negatives
                    mcur += delta_m
                    if np.isnan(mcur) or mcur <= 0:
                        return -np.inf

                return mcur

            #bracket and solve for m0 such that the final mass is equal to the target mass
            #start with reasonable bracket and expand upper bound if needed
            m_low = 1
            m_high = max(target_final_mass, 1e4)

            try:
                #expand m_high until final_mass_from_m0(m_high) >= target_final_mass or until cap
                max_expansions = 100
                exp_count = 0
                while final_mass_from_m0(m_high) < target_final_mass and exp_count < max_expansions:
                    m_high *= 10.0
                    exp_count += 1

                #function must have opposite signs at the bracket ends for brentq
                f_low = final_mass_from_m0(m_low) - target_final_mass
                f_high = final_mass_from_m0(m_high) - target_final_mass

                if f_low == f_high:
                    raise RuntimeError("Cannot bracket root (f_low == f_high).")

                #use brentq to find root
                m0 = brentq(lambda m: final_mass_from_m0(m) - target_final_mass, m_low, m_high,
                            xtol=1e-6, maxiter=200)
            except Exception:
                #keep previous behaviour as a fallback
                m0 = 1

            #produce the SFH using the found m0, same logic as before
            m_current = float(m0)
            sfr_vals = []

            for t in age_centers:
                if (t_d is None) or (tsi <= t_d):  
                    sfr = SFR_ex_situ(t, start_ex, m_current, tau_ex)
                elif (tsi > t_d) and lenticular: 
                    sfr = SFR_in_situ(t, start_ex, m_current, tsi, tau_ex, t_d, tau_c)
                elif (tsi > t_d) and (not lenticular): 
                    sfr = SFR_starburst(t, t_merger, tau_merger_amp, tau_merger_quench, A_burst,
                                        start_ex, m_current, tsi, tau_ex, t_d, tau_c)

                loss = loss_fraction(t)
                mass_added = sfr * dt_years * (1.0 - loss)
                m_current += mass_added

                sfr_vals.append(sfr)

            SFH.append(np.array(sfr_vals))

        return SFH, age_centers
    

    def galaxy_luminosities(
              self
    ):
        def build_ssp_flux_interpolators(ssp_csv, bands):

            csv_path = ssp_csv
            df = pd.read_csv(csv_path)
            zs = np.sort(df['z_obs'].unique())
            first_block = df[df['z_obs'] == zs[0]]
            tage_grid = first_block['tage_gyr'].values

            n_z = len(zs)
            n_t = len(tage_grid)

            mag_columns = [f"mag_{b}" for b in bands]
            mags = df[mag_columns].values.reshape(n_z, n_t, len(bands))
            #magnitudes to fluxes conversion, AB zero point doesn't really matter here since we reconvert later
            fluxes = 10 ** (-0.4 * mags)

            interp_funcs = {}
            for ib, band in enumerate(bands):
                interp_funcs[band] = RegularGridInterpolator(
                    (zs, tage_grid),
                    fluxes[..., ib],
                    bounds_error=False,
                    fill_value=0.0 
                )
            return interp_funcs

        def sfh_list_to_mags(SFH_list, age_centers, z_obs, interp_funcs, bands):
            dt_gyr = np.diff(age_centers).mean()
            dt_years = dt_gyr * 1e9

            t_univ = age_centers.max()
            age_since_formation = t_univ - age_centers

            n_bands = len(bands)
            mags_out = np.full((self.galaxy_number, n_bands), np.inf)

            #for each band flux(tage) interpolated for the whole age array once
            #build query points for the grid: shape (n_time_bins, 2) -> (z_obs, tage)
            query_pts = np.column_stack([np.full_like(age_since_formation, z_obs), age_since_formation])

            #precompute band flux per 1 Msun for each age bin
            band_flux_per_msun = {}
            for ib, band in enumerate(bands):
                f_per_msun = interp_funcs[band](query_pts)  #length n_time_bins
                band_flux_per_msun[band] = f_per_msun  #relative flux units per 1 Msun formed in that bin

            #mass formed per bin (Msun)
            #SFH arrays should be same length as age_centers
            for ig, sfr in enumerate(SFH_list):
                sfr = np.asarray(sfr)

                mass_in_bin = sfr * dt_years

                #sum fluxes per band
                total_fluxes = np.zeros(n_bands)
                for ib, band in enumerate(bands):
                    f_per_msun = band_flux_per_msun[band]         
                    flux_bins = f_per_msun * mass_in_bin        
                    total_flux = flux_bins.sum()
                    total_fluxes[ib] = total_flux

                with np.errstate(divide='ignore'):
                    mags = -2.5 * np.log10(total_fluxes)
                mags[np.isinf(mags)] = np.inf
                mags_out[ig, :] = mags

            return mags_out

        SFH_list, age_centers = self.time_bins_SFH()

        bands = ["u","b","v","sdss_g","sdss_r"]
        interp_funcs = build_ssp_flux_interpolators("scopesim_templates/extragalactic/cluster_mags_z_ages.csv", bands)

        mags_array = sfh_list_to_mags(SFH_list, age_centers, self.z, interp_funcs, bands)
        return mags_array


    def galaxy_velocities(
              self,
              z
    ):

        #this approach assumes a virialized/relaxed cluster, which is generally fine after 1 crossing timescale, which is given for many clusters up to z=1
        #another consideration for the future here: LOS velocities are generally lower toward the center
        #velocity dispersion is generally lower at lower redshift, this should not be significant in a low order model though

        sigma_max = np.sqrt((self.G * self.virial_mass_halo)/self.R_vir(z)[0]) #approximate maximum velocity dispersion of the cluster according to virial theorem in km/s
        v_pec = self.rng.normal(0,sigma_max,self.galaxy_number) #sample peculiar velocities from a gaussian with sigma = v-disp. in km/s
        return v_pec, sigma_max #randomized peculiar velocities, would be assumed here as the LOS peculiar velocities, and cluster velocity dispersio


    def galaxy_effective_radii(
              self,
              m,
              z,
              type
    ):
         
        #as stated in You-Cai Zhang and Xiao-Hu Yang 2019, there seems not to be a clear consensus on environment dependency for the SMR
        #in approximation, the field SMR relation from Van der Wel et al. 2014 (z=0 to z=3) should hold here        
        z_vals  = np.array([0.25, 0.75, 1.25])
        A_early     = 10**np.array([self.rng.normal(0.60, 0.02), self.rng.normal(0.42, 0.01), self.rng.normal(0.22, 0.01)])
        alpha_early = np.array([self.rng.normal(0.75, 0.06), self.rng.normal(0.71, 0.03), self.rng.normal(0.76, 0.04)])
        A_late      = 10**np.array([self.rng.normal(0.86, 0.02), self.rng.normal(0.78, 0.01), self.rng.normal(0.70, 0.01)])
        alpha_late  = np.array([self.rng.normal(0.25, 0.02), self.rng.normal(0.22, 0.02), self.rng.normal(0.22, 0.01)])
        beta_z_early, beta_z_late  = -1.48, -0.75
        log_scatter_early = np.array([0.10, 0.11, 0.12])
        log_scatter_late  = np.array([0.16, 0.16, 0.17])

        if type == "early":
            A_z     = np.interp(z, z_vals, A_early)
            alpha_z = np.interp(z, z_vals, alpha_early)
            beta_z  = beta_z_early
            sigma_z = np.interp(z, z_vals, log_scatter_early)

        else:
            A_z     = np.interp(z, z_vals, A_late)
            alpha_z = np.interp(z, z_vals, alpha_late)
            beta_z  = beta_z_late
            sigma_z = np.interp(z, z_vals, log_scatter_late)

        r_eff_median =  A_z * (1+z)**beta_z * (m / 5e10)**alpha_z #in kpc
        delta_log = self.rng.normal(0.0, sigma_z)  #sigma_z given in dex
        r_eff = r_eff_median * (10.0**delta_log)

        return r_eff


    def galaxy_ellipticities(
            self
    ):


        #fitted relations from the dataset used in Lambas, Maddox, Loveday, 1992, digitized from figure 2
        def fit_elliptical(
                x
         ):
                y = -2554.0783*x**3 + 3386.3684*x**2 - 767.58815*x + 35.83196
                return np.clip(y, 0, None)

        def fit_lenticular(
                x
        ):
                y = -4832.06*x**3 + 8020.53*x**2 - 3744.89*x + 62.445
                return np.clip(y, 0, None)

        def sample_from_fit(
                fit_func
        ):
                x_grid = np.linspace(0, 1, 1000)
                pdf_values = fit_func(x_grid)
                pdf_values /= np.trapz(pdf_values, x_grid)

                cdf = np.cumsum(pdf_values)
                cdf /= cdf[-1]

                u = self.rng.random()
                sample = np.interp(u, cdf, x_grid)
                return sample


        e = np.zeros(self.galaxy_number)
        for i in range(self.galaxy_number):
            if self.activities[i]:
                e[i] = self.rng.uniform(0.2, 0.9)

            elif not self.activities[i] and self.lenticular[i]:
                e[i] = 1 - sample_from_fit(fit_lenticular)

            elif not self.activities[i] and not self.lenticular[i]:
                e[i] = 1 - sample_from_fit(fit_elliptical)

        return e

    def galaxy_angles(
            self
    ):
        return np.array([self.rng.random()*360 for _ in range(self.galaxy_number)])