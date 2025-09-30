from pathlib import Path
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.cosmology import Planck15 as p15
from spextra import Spextrum

from ..rc import Source
from ..extragalactic.cluster_setup import ClusterSetup
from ..extragalactic.galaxies import galaxy, galaxy3d, spiral_two_component, elliptical
from ..extragalactic.galaxy_utils import sersic_profile

def project_positions_to_arcsec(x_mpc, y_mpc, z_cluster):
    D_A = p15.angular_diameter_distance(z_cluster).to(u.Mpc).value
    theta_x = np.array(x_mpc) / D_A
    theta_y = np.array(y_mpc) / D_A
    arcsec_per_rad = (180.0 / np.pi) * 3600.0
    dx_arcsec = theta_x * arcsec_per_rad
    dy_arcsec = theta_y * arcsec_per_rad
    return dx_arcsec, dy_arcsec

def make_empty_image(image_extent_arcsec, pixel_scale_arcsec):
    nx = int(np.ceil(image_extent_arcsec / pixel_scale_arcsec))
    NX_MAX = 20000
    if nx > NX_MAX:
        raise MemoryError(f"Requested image size {nx}x{nx} exceeds safe limit ({NX_MAX}).")
    if nx % 2 == 0:
        nx += 1
    ny = nx
    data = np.zeros((ny, nx), dtype=float)
    wcs_dict = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CRPIX1": (nx + 1) / 2.0,
        "CRPIX2": (ny + 1) / 2.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CDELT1": -1 * (pixel_scale_arcsec / 3600.0),
        "CDELT2": (pixel_scale_arcsec / 3600.0),
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
    }
    wcs = WCS(wcs_dict)
    header = fits.Header(wcs.to_header())
    return data, header, ((nx + 1) / 2.0, (ny + 1) / 2.0)

def place_stamp_on_canvas(canvas, stamp, center_pix, stamp_center_pix, dx_pix, dy_pix):
    ny, nx = canvas.shape
    s_ny, s_nx = stamp.shape
    dst_cx = int(np.round(center_pix[0] + dx_pix))
    dst_cy = int(np.round(center_pix[1] + dy_pix))
    
    x0_dst = dst_cx - int(np.round(stamp_center_pix[0]))
    y0_dst = dst_cy - int(np.round(stamp_center_pix[1]))
    x1_dst = x0_dst + s_nx
    y1_dst = y0_dst + s_ny
    
    x0_clip = max(0, x0_dst)
    y0_clip = max(0, y0_dst)
    x1_clip = min(nx, x1_dst)
    y1_clip = min(ny, y1_dst)
    
    if (x1_clip <= x0_clip) or (y1_clip <= y0_clip):
        return
    
    x0_src = x0_clip - x0_dst
    y0_src = y0_clip - y0_dst
    x1_src = x0_src + (x1_clip - x0_clip)
    y1_src = y0_src + (y1_clip - y0_clip)
    
    canvas[y0_clip:y1_clip, x0_clip:x1_clip] += stamp[y0_src:y1_src, x0_src:x1_src]

def build_cluster_image(
        log_virial_mass_halo,
        z,
        seed,
        N,
        pixel_scale: u.Quantity = 1*u.arcsec,
        image_extent_factor: float = 3.0,
        filter_name: str = "sdss_r",
        use_photometric_scaling: bool = True,
):
    
    """
Generate a simulated image of a galaxy cluster with associated stacked SED.

Parameters
----------
log_virial_mass_halo : float
    Virial mass (M200) of the galaxy cluster in log10(M_sun).
z : float
    Observation redshift of the galaxy cluster. Cannot be 0 and must be smaller than 1.
seed : int
    Seed for numpy random generator for reproducible sampling.
N : int
    Number of galaxies populating the halo. Determined using a HOD model if not specified.
pixel_scale : astropy.units.Quantity, optional
    Pixel scale of the instrument in arcseconds per pixel. Default is 1 arcsec.
image_extent_factor : float, optional
    Multiplier for the cluster virial radius to define the field of view 
    (image extent = 2 * Rvir * extent factor). Default is 3.0.
filter_name : str, optional
    Photometric filter for which the image is generated. Must match one of the 
    filters returned by cluster.galaxy_luminosities(). 
    Current implementation includes Johnson U, B, V and SDSS g and r filters. 
    Default is "sdss_r".
use_photometric_scaling : bool, optional
    If True, scales the intensity for each galaxy stamp by the flux corresponding 
    to its AB magnitude. Default is True.

Returns
-------
Source
    A ScopeSim Source object containing the simulated image (FITS HDU) and the averaged
    SED of all cluster galaxies.
"""

    #build a ClusterSetup instance
    cluster = ClusterSetup(log_virial_mass_halo=log_virial_mass_halo, z=z, seed=seed, N=N)

    #setup SED templates
    elliptical_sed = Spextrum("brown/NGC0584")
    lenticular_sed = Spextrum("brown/NGC5866")
    spiral_sed = Spextrum("brown/NGC3379")
    
    galaxy_seds = []
    a = getattr(cluster, "activities", None)
    l = getattr(cluster, "lenticular", None)
    
    #------- Cluster outputs -------

    #positions
    x3, y3, z3 = cluster.positions
    dx_arcsec, dy_arcsec = project_positions_to_arcsec(x3, y3, cluster.z)
    
    #magnitudes
    filter_names = ["u", "b", "v", "sdss_g", "sdss_r"]
    mags = cluster.galaxy_luminosities()
    filter_idx = filter_names.index(filter_name)
    mags_arr = mags[:, filter_idx]
    
    #effective radii
    eff_r_kpc = cluster.effective_radii
    eff_r_mpc = (eff_r_kpc * u.kpc).to(u.Mpc).value
    D_A = p15.angular_diameter_distance(cluster.z).to(u.Mpc).value
    eff_r_arcsec = (eff_r_mpc / D_A) * (180.0 / np.pi) * 3600.0
    
    #virial radius -> extent
    R_vir = cluster.R_vir_halo
    ang_Rvir_arcsec = (R_vir / D_A) * (180.0 / np.pi) * 3600.0
    image_extent_arcsec = float(2.0 * ang_Rvir_arcsec * image_extent_factor)
    if not np.isfinite(image_extent_arcsec) or image_extent_arcsec <= 0:
        fallback = 3600.0
        print(f"Warning: computed image_extent_arcsec={image_extent_arcsec}. Falling back to {fallback} arcsec.")
        image_extent_arcsec = fallback
    
    pixel_scale_arcsec = float(pixel_scale.to(u.arcsec).value)
    canvas, header, center_pix = make_empty_image(image_extent_arcsec, pixel_scale_arcsec)
    
    #------------- Build each galaxy ---------------
    for idx in range(cluster.galaxy_number):
        m = mags_arr[idx]
        r_eff_arcsec = float(eff_r_arcsec[idx])
        sersic_n = cluster.sersic_indices[idx]
        ell_gal = cluster.ellipticities[idx]
        ang_gal = cluster.angles[idx]
        
        if not a[idx] and not l[idx]:
            sed = elliptical_sed
        elif l[idx]:
            sed = lenticular_sed
        elif a[idx]:
            sed = spiral_sed
        else:
            sed = elliptical_sed
        
        if isinstance(sed, str):
            spec = Spextrum(sed).redshift(z=cluster.z)
        elif isinstance(sed, Spextrum):
            spec = sed.redshift(z=cluster.z)
        galaxy_seds.append(spec)

        stamp = None
        try:
            src = elliptical(
                r_eff_arcsec,
                pixel_scale_arcsec,
                filter_name,
                amplitude=(m * u.ABmag),
                n=sersic_n,
                ellipticity=ell_gal,
                angle=ang_gal
            )
            hdu = getattr(src, 'image_hdu', None)
            if hdu is not None:
                stamp = hdu.data.copy()
            else:
                stamp = src.image.data.copy()
        except Exception:
            #fallback Sersic profile
            try:
                r_eff_pix = max(1, int(np.round(r_eff_arcsec / pixel_scale_arcsec)))
                stamp_size = max(11, int(np.ceil(10 * r_eff_pix)))
                if stamp_size % 2 == 0:
                    stamp_size += 1
                stamp = sersic_profile(
                    r_eff=r_eff_pix,
                    n=sersic_n,
                    ellipticity=ell_gal,
                    angle=ang_gal,
                    normalization="total",
                    width=stamp_size,
                    height=stamp_size
                )
            except Exception:
                continue
        
        if use_photometric_scaling:
            flux_scale = 3631 * 10 ** (-0.4 * float(m))
            stamp *= flux_scale
        
        dx_pix = dx_arcsec[idx] / pixel_scale_arcsec
        dy_pix = dy_arcsec[idx] / pixel_scale_arcsec
        s_ny, s_nx = stamp.shape
        stamp_center = ((s_nx + 1) / 2.0, (s_ny + 1) / 2.0)
        place_stamp_on_canvas(canvas, stamp, center_pix, stamp_center, dx_pix, dy_pix)
    
    #FITS HDU
    header["BUNIT"] = "Jy"
    hdu = fits.PrimaryHDU(data=canvas, header=header)

    #stacking and averaging the spectra
    wave = galaxy_seds[0].wave
    fluxes = np.array([sed.flux for sed in galaxy_seds])
    mean_flux = np.mean(fluxes, axis=0)
    avg_sed = Spextrum((wave, mean_flux))


    src = Source(image_hdu=hdu, spectra=avg_sed)
    return src