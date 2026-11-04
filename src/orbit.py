"""
Étape 1 : géométrie satellite / balise et courbe Doppler théorique.

Hypothèses (simplifications volontaires, à lever plus tard) :
- repère inertiel centré sur la Terre, Terre sphérique et immobile ;
- satellite sur une orbite circulaire polaire, dans le plan x-z ;
- balise fixe à la surface.
"""
import numpy as np

# --- Constantes physiques ---------------------------------------------
MU = 3.986004418e14      # paramètre gravitationnel terrestre [m^3/s^2]
R_EARTH = 6371e3         # rayon terrestre moyen [m]
C = 299_792_458.0        # vitesse de la lumière [m/s]
F0 = 406.0e6             # fréquence d'émission de la balise [Hz]


def orbital_rate(altitude):
    """Vitesse angulaire n = sqrt(mu / a^3) d'une orbite circulaire [rad/s]."""
    a = R_EARTH + altitude
    return np.sqrt(MU / a**3)


def satellite_state(t, altitude=850e3):
    """
    Position et vitesse du satellite à l'instant t (t=0 : passage à
    l'équateur, au-dessus de la longitude 0).
    Retourne (pos, vel), deux tableaux de forme (N, 3) en mètres et m/s.
    """
    t = np.atleast_1d(t).astype(float)
    a = R_EARTH + altitude
    n = orbital_rate(altitude)
    theta = n * t
    pos = a * np.stack([np.cos(theta), np.zeros_like(theta), np.sin(theta)], axis=1)
    vel = a * n * np.stack([-np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=1)
    return pos, vel


def beacon_position(lat_deg, lon_deg):
    """Position (3,) de la balise à la surface, en mètres."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return R_EARTH * np.array([np.cos(lat) * np.cos(lon),
                               np.cos(lat) * np.sin(lon),
                               np.sin(lat)])


def elevation(t, beacon, altitude=850e3):
    """Élévation du satellite vue de la balise [degrés]."""
    pos, _ = satellite_state(t, altitude)
    d = pos - beacon
    up = beacon / np.linalg.norm(beacon)
    sin_el = (d @ up) / np.linalg.norm(d, axis=1)
    return np.degrees(np.arcsin(sin_el))


def range_rate(t, beacon, altitude=850e3):
    """Vitesse de variation de la distance, r_dot [m/s] (formule analytique)."""
    pos, vel = satellite_state(t, altitude)
    d = pos - beacon
    r = np.linalg.norm(d, axis=1)
    return np.sum(d * vel, axis=1) / r


def range_rate_numeric(t, beacon, altitude=850e3):
    """Même grandeur par dérivée numérique : sert à valider la formule analytique."""
    pos, _ = satellite_state(t, altitude)
    r = np.linalg.norm(pos - beacon, axis=1)
    return np.gradient(r, t)


def doppler_shift(t, beacon, altitude=850e3, f0=F0):
    """Décalage Doppler f_reçue - f0 = -f0 * r_dot / c [Hz]."""
    return -f0 * range_rate(t, beacon, altitude) / C


def visibility_window(beacon, altitude=850e3, min_elevation=0.0,
                      half_span=1800.0, dt=1.0):
    """Instants où le satellite est visible (élévation >= min_elevation)."""
    t = np.arange(-half_span, half_span + dt, dt)
    mask = elevation(t, beacon, altitude) >= min_elevation
    return t[mask]