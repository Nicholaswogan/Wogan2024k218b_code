class Star:
    radius : float # in Solar radii
    Teff : float # K
    metal : float # log10(M/H)
    kmag : float
    logg : float # log10 cgs units
    planets : dict # dictionary of planet objects

    def __init__(self, radius, Teff, metal, kmag, logg, planets):
        self.radius = radius
        self.Teff = Teff
        self.metal = metal
        self.kmag = kmag
        self.logg = logg
        self.planets = planets
        
class Planet:
    radius : float # in Earth radii
    mass : float # in Earth masses
    Teq : float # Equilibrium T in K
    transit_duration : float # in seconds
    a: float # semi-major axis in AU
    stellar_constant: float # W/m^2
    
    def __init__(self, radius, mass, Teq, transit_duration, a, stellar_constant):
        self.radius = radius
        self.mass = mass
        self.Teq = Teq
        self.transit_duration = transit_duration
        self.a = a
        self.stellar_constant = stellar_constant

k2_18b = Planet(
    radius=2.610, # Benneke et al. (2019)
    mass=8.63, # Cloutier et al. (2019)
    Teq=278.7, # Matches stellar constant
    transit_duration=2.682*60*60, # Exo.Mast
    a=0.15910, # Benneke et al. (2019)
    stellar_constant=1368.0 # Benneke et al. (2019)
)

k2_18 = Star(
    radius=0.4445, # Benneke et al. (2019)
    Teff=3457, # Benneke et al. (2017)
    metal=0.12, # Exo.Mast
    kmag=8.9, # Exo.Mast
    logg=4.79, # Exo.Mast
    planets={'b':k2_18b}
)








