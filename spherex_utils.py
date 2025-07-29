import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
import glob
import os

def extract_spherex(directory, pattern="level2*cutout*.fits", mask2=None, cr_thresh=10.0):
    """
    Extract spectrophotometry from a directory full of SPHEREx quick release cutouts
    
    Args:
        directory (str):
            Directory where cutouts are stored.
        pattern (str):
            File pattern to search for (default: "level2*cutout*.fits")
        mask2 (`numpy.ndarray`_):
            Boolean mask for pixels where another object lives. Experimental
        cr_thresh (float):
            Cutoff in extracted flux (MJy/sr) above which the exposure is tossed out.
            
    Returns:
        wave (`numpy.ndarray`_):
            Central wavelengths in microns
        dwave (`numpy.ndarray`_):
            FWHM of the corresponding bandpass, in microns
        flux (`numpy.ndarray`_):
            Extracted flux in MJy/sr
        var (`numpy.ndarray`_):
            Variance of the extracted flux, in (MJy/sr)**2
    """
    # Ensure there is a trailing / in the directory string
    if directory[-1] != "/":
        directory = directory + "/"

    # Grab the file names and call the file-based extraction function
    files = glob.glob(directory + pattern)
    
    if not files:
        print(f"Warning: No files found matching pattern '{pattern}' in directory '{directory}'")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    return extract_spherex_from_files(files, mask2=mask2, cr_thresh=cr_thresh)

def extract_spherex_from_files(file_list, mask2=None, cr_thresh=10.0, verbose=False):
    """
    Extract spectrophotometry from a list of SPHEREx cutout files
    
    Args:
        file_list (list):
            List of FITS file paths to process
        mask2 (`numpy.ndarray`_):
            Boolean mask for pixels where another object lives. Experimental
        cr_thresh (float):
            Cutoff in extracted flux (MJy/sr) above which the exposure is tossed out.
        verbose (bool):
            Print detailed processing information
            
    Returns:
        wave (`numpy.ndarray`_):
            Central wavelengths in microns
        dwave (`numpy.ndarray`_):
            FWHM of the corresponding bandpass, in microns
        flux (`numpy.ndarray`_):
            Extracted flux in MJy/sr
        var (`numpy.ndarray`_):
            Variance of the extracted flux, in (MJy/sr)**2
    """
    files = file_list
    flux = np.zeros(len(files))
    wave = np.zeros(len(files))
    dwave = np.zeros(len(files))
    if mask2 is not None:
        flux2 = np.zeros(len(files))
    var = np.zeros(len(files))

    good = np.ones(len(files), dtype=bool)
    processed_count = 0
    size_rejected = 0
    error_count = 0

    for ii in range(len(files)):
        try:
            hdul = fits.open(files[ii])
            img = hdul[1].data
            processed_count += 1
            
            # For now, we only use images where the object is more than 3 pixels away from any edge
            # Expected 6x6 cutout = 36 pixels
            if len(img.flatten()) == 36:
            
                # Object extraction mask, central 2x2 pixels of cutout
                # definitely suboptimal (PSF FWHM can be <1 pixel)
                # but should capture most of the light even at the long wavelength end
                mask = np.zeros_like(img, dtype=bool)
                mask[2:4, 2:4] = True
                mask[np.isnan(img)] = False  # remove phantom pixels

                # Sky mask is everything else
                sky = ~mask
                if mask2 is not None:  # In case there is a second contaminating object
                    sky = sky & ~mask2
                sky[np.isnan(img)] = False  # remove phantom pixels
                
                # Check we have enough sky pixels for background estimation
                if np.sum(sky) < 4:
                    if verbose:
                        print(f"Not enough sky pixels in {os.path.basename(files[ii])}")
                    good[ii] = False
                    continue
                
                # "Aperture photometry"
                # Sum the flux inside the object mask, remove the median sky times the number of object pixels
                sky_median = np.median(img[sky])
                flux[ii] = np.sum(img[mask]) - sky_median * np.sum(mask)
                
                if mask2 is not None:
                    flux2[ii] = np.sum(img[mask2]) - sky_median * np.sum(mask2)
                    
                # Determine wavelength at center of cutout
                # Wavelength map is provided as a 2D interpolation map
                try:
                    wx = hdul[2].data['X'][0]
                    wy = hdul[2].data['Y'][0]
                    wval = hdul[2].data['VALUES'][0][:, :, 0]
                    dwval = hdul[2].data['VALUES'][0][:, :, 1]
                    interp_wave = RegularGridInterpolator((wx, wy), wval)
                    interp_dwave = RegularGridInterpolator((wx, wy), dwval)
                    wave[ii] = interp_wave((3 - hdul[1].header['CRPIX2W'], 3 - hdul[1].header['CRPIX1W']))
                    dwave[ii] = interp_dwave((3 - hdul[1].header['CRPIX2W'], 3 - hdul[1].header['CRPIX1W']))
                except (KeyError, IndexError) as e:
                    if verbose:
                        print(f"Could not extract wavelength from {os.path.basename(files[ii])}: {e}")
                    good[ii] = False
                    continue
                
                # Extremely rough estimate of the variance (full images have a proper variance map, but not cutouts...)
                # First remove the highest and lowest pixels from the sky (sort of like outlier masking)
                if np.sum(sky) > 2:  # Need at least 3 sky pixels
                    sky_values = img[sky]
                    skymask = sky & (img > sky_values.min()) & (img < sky_values.max())
                    
                    if np.sum(skymask) > 1:
                        # Variance of each pixel is first approximated by variance of sky pixels
                        var[ii] = np.sum(mask) * np.var(img[skymask])
                        # Now we adjust the variance higher assuming that variance is proportional to flux
                        # but we only include the pixels above the sky background to avoid decreasing it
                        posmask = img[mask] > sky_median
                        if np.sum(posmask) > 0:
                            var[ii] *= 1 + np.sum(((img[mask] - sky_median) / sky_median)[posmask])
                    else:
                        var[ii] = np.sum(mask) * np.var(sky_values)
                else:
                    good[ii] = False
                    continue
                
            else:
                good[ii] = False
                size_rejected += 1
                if verbose:
                    print(f"Wrong image size ({len(img.flatten())} pixels): {os.path.basename(files[ii])}")
                
            hdul.close()
            
        except Exception as e:
            if verbose:
                print(f"Error processing file {os.path.basename(files[ii])}: {e}")
            good[ii] = False
            error_count += 1
    
    # Remove any fluxes that are "bad" for some reason.
    # Default value of cr_thresh is good for faint high-z objects
    # Set it to a higher value if your object is super bright
    flux_good = (~np.isnan(flux)) & (~np.isnan(wave)) & (np.abs(flux) < cr_thresh)
    final_good = good & flux_good
    
    if verbose:
        print(f"Processing summary:")
        print(f"  Files processed: {processed_count}/{len(files)}")
        print(f"  Files with errors: {error_count}")
        print(f"  Files rejected (wrong size): {size_rejected}")
        print(f"  Files rejected (bad flux/wave): {np.sum(good) - np.sum(final_good)}")
    
    print(f"{np.sum(final_good)} out of {len(files)} exposures are good.")
    
    wave = wave[final_good]
    dwave = dwave[final_good]
    flux = flux[final_good]
    var = var[final_good]

    if mask2 is None:
        return wave, dwave, flux, var
    else:
        flux2 = flux2[final_good]
        return wave, dwave, flux, var, flux2

def get_flam(wave, flux, omega_pix=37.932):
    """
    Convert MJy/sr to f_lambda in cgs units
    
    Args:
        wave (array): Wavelength in microns
        flux (array): Flux in MJy/sr
        omega_pix (float): Pixel solid angle in arcsec^2
        
    Returns:
        flam (array): Flux in erg/s/cm²/Å
    """
    nu = 2.998e8 / (wave * 1e-6)  # Frequency in Hz
    flam = flux * (1e6 * 1e-23) * (nu / (wave * 1e4)) / (u.sr.to(u.arcsec * u.arcsec)) * omega_pix
    return flam
    
def get_fnu_from_flam(wave, flam):
    """
    Convert f_lambda to f_nu
    
    Args:
        wave (array): Wavelength in microns
        flam (array): f_lambda in cgs units
        
    Returns:
        fnu (array): f_nu in mJy
    """
    nu = 2.998e8 / (wave * 1e-6)  # Frequency in Hz
    fnu = flam * ((wave * 1e4) / nu)  # Convert to f_nu in cgs
    return 1000 * fnu * 1e23  # Convert to mJy
    
def rebin_spherex(wave, flam, std, dwave, tol=0.1):
    """
    Simple rebinning scheme for SPHEREx spectra
    
    Args:
        wave (array): Wavelength array
        flam (array): Flux array
        std (array): Error array
        dwave (array): Wavelength bin width array
        tol (float): Tolerance for grouping nearby wavelengths
        
    Returns:
        tuple: Rebinned wave, flam, std, dwave arrays
    """
    sort = np.argsort(wave)
    wave = wave[sort]
    flam = flam[sort]
    std = std[sort]
    dwave = dwave[sort]
    used = np.zeros_like(wave)
    ii = 0
    wave_bin = []
    flam_bin = []
    std_bin = []
    dwave_bin = []
    
    while ii < len(wave):
        close = (np.abs(wave - wave[ii]) < tol * dwave[ii]) & (used == 0)
        wave_bin.append(np.mean(wave[close]))
        flam_bin.append(np.mean(flam[close]))
        std_bin.append(np.sqrt(np.sum(std[close]**2) / np.sum(close)**2))
        dwave_bin.append(np.max(wave[close] + 0.5 * dwave[close]) - np.min(wave[close] - 0.5 * dwave[close]))
        used[close] = 1
        ii = np.arange(len(wave))[close][-1] + 1

    return np.array(wave_bin), np.array(flam_bin), np.array(std_bin), np.array(dwave_bin)

# Common emission lines for plotting (rest wavelengths in Angstroms)
line_list = [6564.6, 5008.2, 4862.7, 2800.0, 1908.7, 1550.0, 1215.67]
line_names = ['Hα', '[OIII]', 'Hβ', 'MgII', 'CIII]', 'CIV', 'Lyα']

def plot_spherex_flam(wave, flam, std, dwave, zqso=None, label=None, wave_old=None, flam_old=None, flam2=None, 
                     xlim=(0.65, 5.05), figsize=(9, 5), show_lines=True, line_labels=False):
    """
    Plot SPHEREx spectrum in f_lambda units
    
    Args:
        wave (array): Wavelength in microns
        flam (array): f_lambda flux in cgs
        std (array): Error in f_lambda
        dwave (array): Wavelength bin widths
        zqso (float, optional): Redshift for emission line markers
        label (str, optional): Label for the spectrum
        wave_old (array, optional): Comparison wavelength array
        flam_old (array, optional): Comparison flux array
        flam2 (array, optional): Second object flux array
        xlim (tuple): Wavelength limits for plot
        figsize (tuple): Figure size
        show_lines (bool): Show emission line markers
        line_labels (bool): Label emission lines
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.errorbar(wave, flam, c='k', fmt='o', xerr=0.5*dwave, yerr=std, label=label, ms=3.5)
    
    if flam2 is not None:
        plt.errorbar(wave, flam2, c='r', fmt='x', xerr=0.5*dwave, label='other', ms=4)
    
    if flam_old is not None:
        plt.plot(wave_old, flam_old, c='k', lw=0.15, alpha=0.15)
    
    plt.axhline(0.0, c='darkorange', lw=0.5, linestyle='dashed')
    
    if label:
        plt.legend(fontsize=16)
    
    plt.ylim(-0.2*flam.max(), 1.2*flam.max())
    plt.xlim(xlim[0], xlim[1])
    
    if zqso is not None and show_lines:
        for i, line in enumerate(line_list):
            line_wave = line * (1 + zqso) * 1e-4  # Convert to microns
            if xlim[0] <= line_wave <= xlim[1]:
                plt.axvline(line_wave, c='k', linestyle='dotted', alpha=0.7)
                if line_labels:
                    plt.text(line_wave, 0.9*flam.max(), line_names[i], 
                           rotation=90, ha='center', va='bottom', fontsize=10)
    
    plt.ylabel('F$_λ$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]', fontsize=18)
    plt.xlabel('Wavelength [μm]', fontsize=18)
    plt.tick_params(which='both', top=True, right=True, direction='in', labelsize=13)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

def plot_spherex_fnu(wave, fnu, std, dwave, zqso=None, label=None, wave_old=None, fnu_old=None, fnu2=None,
                    xlim=(0.65, 5.05), figsize=(9, 5), show_lines=True, line_labels=False):
    """
    Plot SPHEREx spectrum in f_nu units
    
    Args:
        wave (array): Wavelength in microns
        fnu (array): f_nu flux in mJy
        std (array): Error in f_nu
        dwave (array): Wavelength bin widths
        zqso (float, optional): Redshift for emission line markers
        label (str, optional): Label for the spectrum
        wave_old (array, optional): Comparison wavelength array
        fnu_old (array, optional): Comparison flux array
        fnu2 (array, optional): Second object flux array
        xlim (tuple): Wavelength limits for plot
        figsize (tuple): Figure size
        show_lines (bool): Show emission line markers
        line_labels (bool): Label emission lines
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.errorbar(wave, fnu, c='k', fmt='o', xerr=0.5*dwave, yerr=std, label=label, ms=3.5)
    
    if fnu2 is not None:
        plt.errorbar(wave, fnu2, c='r', fmt='x', xerr=0.5*dwave, label='other', ms=4)
    
    if fnu_old is not None:
        plt.plot(wave_old, fnu_old, c='k', lw=0.15, alpha=0.15)
    
    plt.axhline(0.0, c='darkorange', lw=0.5, linestyle='dashed')
    
    if label:
        plt.legend(fontsize=16, loc='upper left')
    
    plt.ylim(-0.2*fnu.max(), 1.2*fnu.max())
    plt.xlim(xlim[0], xlim[1])
    
    if zqso is not None and show_lines:
        for i, line in enumerate(line_list):
            line_wave = line * (1 + zqso) * 1e-4  # Convert to microns
            if xlim[0] <= line_wave <= xlim[1]:
                plt.axvline(line_wave, c='k', linestyle='dotted', alpha=0.7)
                if line_labels:
                    plt.text(line_wave, 0.9*fnu.max(), line_names[i], 
                           rotation=90, ha='center', va='bottom', fontsize=10)
    
    plt.ylabel('F$_ν$ [mJy]', fontsize=18)
    plt.xlabel('Wavelength [μm]', fontsize=18)
    plt.tick_params(which='both', top=True, right=True, direction='in', labelsize=13)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

def get_spherex_info(wave, flux, var):
    """
    Get basic information about a SPHEREx spectrum
    
    Args:
        wave (array): Wavelength array
        flux (array): Flux array
        var (array): Variance array
        
    Returns:
        dict: Dictionary with spectrum information
    """
    std = np.sqrt(var)
    snr = flux / std
    
    info = {
        'n_points': len(wave),
        'wave_min': np.min(wave),
        'wave_max': np.max(wave),
        'wave_range': np.max(wave) - np.min(wave),
        'median_flux': np.median(flux),
        'median_snr': np.median(snr),
        'max_snr': np.max(snr),
        'positive_flux_fraction': np.sum(flux > 0) / len(flux)
    }
    
    return info