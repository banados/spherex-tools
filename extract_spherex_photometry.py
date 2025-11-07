#!/usr/bin/env python
"""
Extract and plot SPHEREx photometry from cutout files
"""

import argparse
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

# Import the SPHEREx utilities (assumes spherex_utils.py is in the same directory)
try:
    from spherex_utils import get_flam, get_fnu_from_flam, plot_spherex_flam, plot_spherex_fnu
except ImportError:
    print("Error: spherex_utils.py not found in current directory")  
    print("Please make sure spherex_utils.py is in the same directory as this script")
    sys.exit(1)

def extract_spherex_from_files(file_list, mask2=None, cr_thresh=10.0):
    """
    Modified version of extract_spherex that works with a provided file list
    instead of searching for files with a hardcoded pattern
    
    Args:
        file_list (list): List of FITS file paths to process
        mask2 (`numpy.ndarray`_): Boolean mask for pixels where another object lives. Experimental
        cr_thresh (float): Cutoff in extracted flux (MJy/sr) above which the exposure is tossed out.
            
    Returns:
        wave (`numpy.ndarray`_): Central wavelengths in microns
        dwave (`numpy.ndarray`_): FWHM of the corresponding bandpass, in microns
        flux (`numpy.ndarray`_): Extracted flux in MJy/sr
        var (`numpy.ndarray`_): Variance of the extracted flux, in (MJy/sr)**2
    """
    from astropy.io import fits
    from scipy.interpolate import RegularGridInterpolator
    
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
            img = hdul['IMAGE'].data
            processed_count += 1
            
            # For now, we only use images where the object is more than 3 pixels away from any edge
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
                
                # "Aperture photometry"
                # Sum the flux inside the object mask, remove the median sky times the number of object pixels
                flux[ii] = np.sum(img[mask]) - np.median(img[sky]) * (np.sum(mask))
                if mask2 is not None:
                    flux2[ii] = np.sum(img[mask2]) - np.median(img[sky]) * (np.sum(mask2))
                    
                # Determine wavelength at center of cutout
                # Wavelength map is provided as a 2D interpolation map
                
                wx = hdul['WCS-WAVE'].data['X']
                if len(wx.shape)==1:
                    wy = hdul['WCS-WAVE'].data['Y']
                    wval = hdul['WCS-WAVE'].data['VALUES'][:, :, 0]
                    dwval = hdul['WCS-WAVE'].data['VALUES'][:, :, 1]
                else:
                    wx = hdul['WCS-WAVE'].data['X'][0]
                    wy = hdul['WCS-WAVE'].data['Y'][0]
                    wval = hdul['WCS-WAVE'].data['VALUES'][0][:, :, 0]
                    dwval = hdul['WCS-WAVE'].data['VALUES'][0][:, :, 1]

                if wval.shape[0] != len(wx):
                    wval = wval.T
                    dwval = dwval.T
                    interp_wave = RegularGridInterpolator((wx, wy), wval)
                    interp_dwave = RegularGridInterpolator((wx, wy), dwval)
                    wave[ii] = interp_wave((3 - hdul['IMAGE'].header['CRPIX1W'], 3 - hdul['IMAGE'].header['CRPIX2W']))
                    dwave[ii] = interp_dwave((3 - hdul['IMAGE'].header['CRPIX1W'], 3 - hdul['IMAGE'].header['CRPIX2W']))
                else:
                    interp_wave = RegularGridInterpolator((wx, wy), wval)
                    interp_dwave = RegularGridInterpolator((wx, wy), dwval)
                    wave[ii] = interp_wave((3 - hdul['IMAGE'].header['CRPIX2W'], 3 - hdul['IMAGE'].header['CRPIX1W']))
                    dwave[ii] = interp_dwave((3 - hdul['IMAGE'].header['CRPIX2W'], 3 - hdul['IMAGE'].header['CRPIX1W']))
                
                # Extremely rough estimate of the variance (full images have a proper variance map, but not cutouts...)
                # First remove the highest and lowest pixels from the sky (sort of like outlier masking)
                #skymask = sky & (img > img[sky].min()) & (img < img[sky].max())
                # Variance of each pixel is first approximated by variance of sky pixels
                var[ii] =  np.sum(mask*hdul['VARIANCE'].data)
                # Now we adjust the variance higher assuming that variance is proportional to flux
                # but we only include the pixels above the sky background to avoid decreasing it
                #posmask = img[mask] > np.median(img[sky])
                #var[ii] *= 1 + np.sum(((img[mask] - np.median(img[sky])) / np.median(img[sky]))[posmask])
                
            else:
                good[ii] = False
                size_rejected += 1
                
            hdul.close()
            
        except Exception as e:
            print(f"Error processing file {os.path.basename(files[ii])}: {e}")
            good[ii] = False
            error_count += 1
    
    # Remove any fluxes that are "bad" for some reason.
    # Default value of cr_thresh is good for faint high-z objects
    # Set it to a higher value if your object is super bright
    flux_good = (~np.isnan(flux)) & (~np.isnan(wave)) & (np.abs(flux) < cr_thresh)
    final_good = good & flux_good
    
    print(f"Processing summary:")
    print(f"  Files processed: {processed_count}/{len(files)}")
    print(f"  Files with errors: {error_count}")
    print(f"  Files rejected (wrong size): {size_rejected}")
    print(f"  Files rejected (bad flux/wave): {np.sum(good) - np.sum(final_good)}")
    print(f"  Final good exposures: {np.sum(final_good)}")
    
    wave = wave[final_good]
    dwave = dwave[final_good]
    flux = flux[final_good]
    var = var[final_good]

    if mask2 is None:
        return wave, dwave, flux, var
    else:
        flux2 = flux2[final_good]
        return wave, dwave, flux, var, flux2

def load_quasar_template(template_name, redshift=0.0):
    """
    Load and process a quasar template for overplotting
    
    Parameters:
    -----------
    template_name : str
        Name of the template to load ('vandenberk01')
    redshift : float
        Redshift to apply to the template
        
    Returns:
    --------
    wave_template : array
        Template wavelengths in microns (observer frame)
    flux_template : array  
        Template flux density (relative units)
    """
    
    if template_name.lower() == 'vandenberk01':
        # Try to find the Vanden Berk template file in multiple locations
        template_files = [
            'templates/sdss_qso_vandenberk_fitting.txt',  # User's suggested location
            'sdss_qso_vandenberk_fitting.txt',            # Current directory
            'templates/vandenberk01_template.txt',
            'vandenberk01_template.txt',
            'templates/vandenberk_template.txt',
            'vandenberk_template.txt'
        ]
        
        template_file = None
        for fname in template_files:
            if os.path.exists(fname):
                template_file = fname
                break
        
        if template_file is None:
            print(f"Warning: Could not find Vanden Berk template file.")
            print(f"Looked for: {', '.join(template_files)}")
            print(f"Please ensure the template file exists in one of these locations.")
            return None, None
            
        try:
            # Load the template data
            # Format: wavelength(Å), flux, error
            data = np.loadtxt(template_file)
            wave_angstrom = data[:, 0]  # Wavelength in Angstroms (rest-frame)
            flux_template = data[:, 1]  # Flux density (relative units)
            
            # Apply redshift: λ_observed = λ_rest × (1+z)
            wave_angstrom_obs = wave_angstrom * (1.0 + redshift)
            
            # Convert to microns: λ_micron = λ_angstrom / 10000
            wave_template = wave_angstrom_obs / 10000.0
            
            print(f"Loaded Vanden Berk template from {template_file}")
            print(f"Template has {len(wave_template)} points")
            print(f"Rest-frame wavelength range: {wave_angstrom.min():.1f} - {wave_angstrom.max():.1f} Å")
            print(f"Observer-frame wavelength range: {wave_template.min():.3f} - {wave_template.max():.3f} μm")
            
            return wave_template, flux_template
            
        except Exception as e:
            print(f"Error loading template file {template_file}: {e}")
            return None, None
    
    else:
        print(f"Unknown template: {template_name}")
        print("Available templates: vandenberk01")
        return None, None

def scale_template_to_data(wave_data, flux_data, wave_template, flux_template):
    """
    Scale template to roughly match the data flux levels
    
    Parameters:
    -----------
    wave_data, flux_data : arrays
        Observed spectrum
    wave_template, flux_template : arrays
        Template spectrum
        
    Returns:
    --------
    flux_template_scaled : array
        Template flux scaled to match data
    """
    
    # Find overlapping wavelength range
    wave_min = max(wave_data.min(), wave_template.min(), 2.1)#, 3.0)
    wave_max = min(wave_data.max(), wave_template.max())
    
    if wave_min >= wave_max:
        print("Warning: No wavelength overlap between data and template")
        return flux_template
    
    # Get median flux in overlapping region
    data_mask = (wave_data >= wave_min) & (wave_data <= wave_max)
    template_mask = (wave_template >= wave_min) & (wave_template <= wave_max)
    
    if np.sum(data_mask) == 0 or np.sum(template_mask) == 0:
        print("Warning: No overlapping data points for scaling")
        return flux_template
    
    # Use median flux for scaling, but only consider positive flux values
    data_flux_positive = flux_data[data_mask & (flux_data > 0)]
    template_flux_positive = flux_template[template_mask & (flux_template > 0)]
    
    if len(data_flux_positive) > 0 and len(template_flux_positive) > 0:
        median_data = np.median(data_flux_positive)
        median_template = np.median(template_flux_positive)
        
        if median_template > 0:
            scale_factor = median_data / median_template
            print(f"Scaling template by factor {scale_factor:.2e}")
            return flux_template * scale_factor
    
    print("Warning: Could not determine proper scaling, using template as-is")
    return flux_template

def convert_flam_to_fnu(wave_micron, flam):
    """
    Convert f_lambda to f_nu
    
    Parameters:
    -----------
    wave_micron : array
        Wavelength in microns
    flam : array
        Flux in f_lambda units (erg/s/cm²/Å)
        
    Returns:
    --------
    fnu : array
        Flux in f_nu units (mJy)
    """
    # Convert wavelength to cm
    wave_cm = wave_micron * 1e-4
    
    # Speed of light in cm/s
    c = 2.998e10
    
    # Convert: f_nu = f_lambda * lambda^2 / c
    # Factor of 1e26 converts from erg/s/cm²/Hz to mJy
    fnu = flam * wave_cm**2 / c * 1e26
    
    return fnu

def rebin_spectrum_spherex(wave, flux, std, dwave, tolerance=0.1):
    """
    Rebin spectrum using the SPHEREx-specific rebinning from spherex_utils
    
    Parameters:
    -----------
    wave : array
        Wavelength array in microns
    flux : array  
        Flux array
    std : array
        Error array (same units as flux)
    dwave : array
        Wavelength bin width array from SPHEREx
    tolerance : float
        Tolerance for grouping nearby wavelengths (default: 0.1 = 10%)
        
    Returns:
    --------
    wave_rebinned : array
        Rebinned wavelength array
    flux_rebinned : array
        Rebinned flux array  
    std_rebinned : array
        Rebinned error array (proper error propagation)
    dwave_rebinned : array
        Rebinned wavelength bin widths
    """
    try:
        from spherex_utils import rebin_spherex
        
        # Use the existing SPHEREx rebinning function
        wave_rebinned, flux_rebinned, std_rebinned, dwave_rebinned = rebin_spherex(
            wave, flux, std, dwave, tol=tolerance)
        
        print(f"SPHEREx rebinning with tolerance {tolerance:.1f} ({tolerance*100:.0f}% of bandwidth)")
        print(f"Original → Rebinned points: {len(wave)} → {len(wave_rebinned)}")
        print(f"Wavelength range: {wave_rebinned.min():.4f} - {wave_rebinned.max():.4f} μm")
        print(f"Mean wavelength spacing: {np.mean(np.diff(wave_rebinned)):.4f} μm")
        
        return wave_rebinned, flux_rebinned, std_rebinned, dwave_rebinned
        
    except ImportError:
        raise ImportError("spherex_utils not available, cannot use SPHEREx rebinning")
    except Exception as e:
        raise RuntimeError(f"Could not rebin spectrum: {e}")

def rebin_spectrum(wave, flux, std, delta_wave, plot_type='flam'):
    """
    Rebin spectrum to a specified wavelength spacing (uniform grid)
    
    Parameters:
    -----------
    wave : array
        Wavelength array in microns
    flux : array  
        Flux array
    std : array
        Error array (same units as flux) - not used, we calculate std from binned data
    delta_wave : float
        Desired wavelength spacing in microns
    plot_type : str
        Either 'flam' or 'fnu' for flux units (for info only)
        
    Returns:
    --------
    wave_rebinned : array
        Rebinned wavelength array (bin centers)
    flux_rebinned : array
        Rebinned flux array (mean of values in each bin)
    std_rebinned : array
        Standard deviation of flux values in each bin
    actual_delta_wave : float
        Actual wavelength spacing used
    """
    
    # Sort data by wavelength (keep all points, including duplicates)
    sort_indices = np.argsort(wave)
    wave_sorted = wave[sort_indices]
    flux_sorted = flux[sort_indices]
    
    if len(wave_sorted) < 2:
        raise ValueError(f"Not enough wavelength points for rebinning")
    
    # Create wavelength bins
    wave_min = wave_sorted.min()
    wave_max = wave_sorted.max()
    
    # Create bin edges
    bin_edges = np.arange(wave_min, wave_max + delta_wave, delta_wave)
    n_bins = len(bin_edges) - 1
    
    if n_bins < 2:
        raise ValueError(f"Delta wavelength {delta_wave:.4f} μm is too large for wavelength range {wave_max-wave_min:.4f} μm")
    
    # Initialize output arrays
    wave_rebinned = []
    flux_rebinned = []
    std_rebinned = []
    
    # Bin the data (including all measurements, even at same wavelengths)
    for i in range(n_bins):
        # Find points in this bin
        in_bin = (wave_sorted >= bin_edges[i]) & (wave_sorted < bin_edges[i+1])
        
        if np.sum(in_bin) > 0:
            # Calculate bin center, mean flux, and standard deviation
            wave_center = (bin_edges[i] + bin_edges[i+1]) / 2.0
            flux_in_bin = flux_sorted[in_bin]
            
            mean_flux = np.mean(flux_in_bin)
            if len(flux_in_bin) > 1:
                std_flux = np.std(flux_in_bin, ddof=1)  # Sample standard deviation
            else:
                std_flux = 0.0  # Single point in bin
            
            wave_rebinned.append(wave_center)
            flux_rebinned.append(mean_flux)
            std_rebinned.append(std_flux)
    
    wave_rebinned = np.array(wave_rebinned)
    flux_rebinned = np.array(flux_rebinned)
    std_rebinned = np.array(std_rebinned)
    
    if len(wave_rebinned) == 0:
        raise ValueError(f"No valid bins created with delta wavelength {delta_wave:.4f} μm")
    
    # Calculate actual delta wavelength used
    actual_delta_wave = delta_wave
    
    # Count total measurements and unique wavelengths for info
    unique_waves = len(np.unique(wave_sorted))
    total_measurements = len(wave_sorted)
    
    print(f"Uniform grid rebinning to Δλ = {delta_wave:.4f} μm: {len(wave_rebinned)} bins")
    print(f"Total measurements: {total_measurements} (including {total_measurements - unique_waves} repeated wavelengths)")
    print(f"Original → Rebinned points: {total_measurements} → {len(wave_rebinned)}")
    print(f"Wavelength range: {wave_rebinned.min():.4f} - {wave_rebinned.max():.4f} μm")
    
    return wave_rebinned, flux_rebinned, std_rebinned, actual_delta_wave

EXAMPLES = """
Examples:

# Extract photometry for an object with cutouts in a specific directory
python extract_spherex_photometry.py --name NGC6888 --directory cutouts/

# Specify redshift for emission line markers and quasar template overlay
python extract_spherex_photometry.py --name J0916-2511 --directory data/ --redshift 4.85 --show_template vandenberk01

# Save plots and spectrum data to files
python extract_spherex_photometry.py --name M31 --directory spherex_data/ --save_plots --save_spectrum

# Custom output directory for plots and data
python extract_spherex_photometry.py --name Crab --directory data/ --save_plots --save_spectrum --output_dir results/

# Both flux density units with saved data
python extract_spherex_photometry.py --name NGC2024 --directory cutouts/ --plot_fnu --plot_flam --save_spectrum

# Plot high-redshift quasar with template overlay
python extract_spherex_photometry.py --name E273+65 --directory quasars_cutouts/ --redshift 5.4 --show_template vandenberk01

# Show rebinned spectrum using SPHEREx method (10% tolerance)
python extract_spherex_photometry.py --name NGC1068 --directory cutouts/ --show_rebin_spherex 0.1

# Show rebinned spectrum using uniform grid (0.1 μm spacing)
python extract_spherex_photometry.py --name NGC1068 --directory cutouts/ --show_rebin_uniform 0.1

# Full analysis with template and SPHEREx rebinning (5% tolerance)
python extract_spherex_photometry.py --name J1030+0524 --directory quasars/ --redshift 6.3 --show_template vandenberk01 --show_rebin_spherex 0.05 --save_plots --save_spectrum

# Full analysis with template and uniform grid rebinning (0.05 μm spacing)
python extract_spherex_photometry.py --name E273+65 --directory quasars_cutouts/ --redshift 5.4 --show_template vandenberk01 --show_rebin_uniform 0.05 --save_plots --save_spectrum

Expected directory structure:
cutouts/
├── NGC6888_cutout_level2_2025W19_1B_0501_2D5_spx_l2b-v11-2025-162.fits
├── NGC6888_cutout_level2_2025W19_1B_0501_2D2_spx_l2b-v11-2025-162.fits
└── ...

Template files (place in templates/ directory or current directory):
templates/sdss_qso_vandenberk_fitting.txt

Output files:
├── NGC6888_spherex_flam.png/pdf (plots)
├── NGC6888_spherex_fnu.png/pdf (plots)  
├── NGC6888_spherex_spectrum_flam.txt (original spectral data)
├── NGC6888_spherex_spectrum_rebinned_spherex_0.10_flam.txt (SPHEREx rebinned)
├── NGC6888_spherex_spectrum_rebinned_uniform_0.100_flam.txt (uniform grid rebinned)
└── NGC6888_spherex_spectrum_fnu.txt (original spectral data)
"""

def find_cutout_files(name, directory):
    """
    Find all SPHEREx cutout files for a given object name
    
    Parameters:
    -----------
    name : str
        Object name to search for
    directory : str
        Directory containing cutout files
    
    Returns:
    --------
    files : list
        List of matching cutout files
    """
    # Ensure directory has trailing slash
    if not directory.endswith('/'):
        directory += '/'
    
    # Search patterns - try multiple possible naming conventions
    search_patterns = [
        f"{name}_cutout_level2*.fits",           # From our downloader script
        f"{name}_level2*.fits",                  # Without "cutout" in name
        f"*{name}*cutout*.fits",                 # Name anywhere in filename
        f"*{name}*level2*.fits"                  # More general pattern
    ]
    
    files = []
    for pattern in search_patterns:
        matches = glob.glob(os.path.join(directory, pattern))
        files.extend(matches)
    
    # Remove duplicates while preserving order
    files = list(dict.fromkeys(files))
    
    return files

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

def plot_spherex_spectrum(wave, flam, std, dwave, name, redshift=None, 
                         save_plots=False, output_dir='.', plot_type='flam',
                         template_data=None, rebin_method=None, rebin_param=None):
    """
    Create a SPHEREx spectrum plot using the spherex_utils plotting functions
    
    Parameters:
    -----------
    wave : array
        Wavelength array in microns
    flam : array
        Flux in f_lambda units (cgs) or f_nu units (mJy)
    std : array
        Error array in same units as flam
    dwave : array
        Wavelength bin widths
    name : str
        Object name for plot label
    redshift : float, optional
        Redshift for emission line markers
    save_plots : bool
        If True, save plot to file instead of displaying
    output_dir : str
        Directory to save plots
    plot_type : str
        Either 'flam' or 'fnu' for flux units
    template_data : tuple, optional
        (wave_template, flux_template) for overplotting template
    rebin_method : str, optional
        Rebinning method used ('spherex' or 'uniform')
    rebin_param : float, optional
        Rebinning parameter (tolerance for SPHEREx, delta_wave for uniform)
    template_data : tuple, optional
        (wave_template, flux_template) for overplotting template
    show_rebin : float, optional
        If provided, show rebinned spectrum with this delta wavelength (in microns) instead of original data points
    """
    
    # Create label
    if redshift is not None:
        label = f"{name}, z={redshift:.2f}"
    else:
        label = name
    
    # Create the plot
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    
    # Plot the data (already rebinned if requested)
    if rebin_method is not None:
        if rebin_method == "spherex":
            label_suffix = f"SPHEREx rebin (tol={rebin_param:.2f})"
        elif rebin_method == "uniform":
            label_suffix = f"Δλ={rebin_param:.3f}μm"
        plot_label = f'{label} ({label_suffix})'
    else:
        plot_label = label
    
    plt.errorbar(wave, flam, c='k', fmt='o', xerr=0.5*dwave, yerr=std, 
                label=plot_label, ms=3.5, zorder=3)
    plt.axhline(0.0, c='darkorange', lw=0.5, linestyle='dashed')

    # Plot template if provided (higher zorder so it's visible above data)
    if template_data is not None:
        wave_template, flux_template = template_data
        
        # Note: template is now scaled to the data being plotted (rebinned if applicable)
        if rebin_method is not None:
            print(f"Scaling template to rebinned data ({rebin_method} method)")
        
        if plot_type == 'flam':
            # Scale template to match data in f_lambda units
            flux_template_scaled = scale_template_to_data(wave, flam, wave_template, flux_template)
        elif plot_type == 'fnu':
            # Convert template to f_nu and then scale
            flux_template_fnu = convert_flam_to_fnu(wave_template, flux_template)
            flux_template_scaled = scale_template_to_data(wave, flam, wave_template, flux_template_fnu)
        
        plt.plot(wave_template, flux_template_scaled, 'r-', alpha=0.8, lw=1.1, 
                label='Vanden Berk Template', zorder=5)
    
    # Add legend if there are labels
    if plot_label or template_data is not None:
        plt.legend(fontsize=16, loc='upper left')
    
    # Set plot limits using the actual data being plotted
    if np.median(flam) > 0:
        plt.ylim(-3*np.median(flam), 20*np.median(flam))
    else:
        plt.ylim(2*np.median(flam), -10*np.median(flam))

    plt.xlim(0.65, 5.05)
    
    # Add emission line markers if redshift is provided
    if redshift is not None:
        # Common emission lines in Angstroms (rest frame)
        line_list = [6564.6, 5008.2, 4862.7, 2800.0, 1908.7, 1550.0, 1215.67]
        line_names = ['Hα', '[OIII]', 'Hβ', 'MgII', 'CIII]', 'CIV', 'Lyα']
        
        for line_wave, line_name in zip(line_list, line_names):
            line_wave_obs = line_wave * (1 + redshift) * 1e-4  # Convert to microns, observer frame
            if 0.65 <= line_wave_obs <= 5.05:
                plt.axvline(line_wave_obs, c='gray', linestyle='dotted', alpha=0.7, zorder=1)
                # Add line label at top of plot
                plt.text(line_wave_obs, 0.95*plt.ylim()[1], line_name, 
                        rotation=90, ha='right', va='top', fontsize=8, alpha=0.7)
    
    # Set labels and formatting
    if plot_type == 'flam':
        plt.ylabel('F$_λ$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]', fontsize=18)
    elif plot_type == 'fnu':
        plt.ylabel('F$_ν$ [mJy]', fontsize=18)
    
    plt.xlabel('Wavelength [μm]', fontsize=18)
    plt.tick_params(which='both', top=True, right=True, direction='in', labelsize=13)
    plt.minorticks_on()
    plt.tight_layout()
    
    if save_plots:
        # Save both PNG and PDF
        png_path = os.path.join(output_dir, f"{name}_spherex_{plot_type}.png")
        pdf_path = os.path.join(output_dir, f"{name}_spherex_{plot_type}.pdf")
        
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved {plot_type} plot: {png_path}")
        print(f"Saved {plot_type} plot: {pdf_path}")
        plt.close()  # Close the figure to free memory
    else:
        plt.show()

def save_spectrum_to_file(wave, flux, error, name, output_dir='.', flux_type='flam', rebin_info=None):
    """
    Save spectrum to a text file
    
    Parameters:
    -----------
    wave : array
        Wavelength array in microns
    flux : array
        Flux array
    error : array
        Error array (same units as flux)
    name : str
        Object name for filename
    output_dir : str
        Directory to save file
    flux_type : str
        Type of flux units for filename ('flam' or 'fnu')
    rebin_info : tuple, optional
        If provided, tuple of (method, parameter) for rebinned filename and header
    flux_type : str
        Type of flux units for filename ('flam' or 'fnu')
    rebin_info : tuple, optional
        If provided, tuple of (method, parameter) for rebinned filename and header
    flux_type : str
        Type of flux units for filename ('flam' or 'fnu')
    rebin_info : tuple, optional
        If provided, tuple of (method, parameter) for rebinned filename and header
    """
    if rebin_info is not None:
        method, param = rebin_info
        if method == "spherex":
            filename = f"{name}_spherex_spectrum_rebinned_spherex_{param:.2f}_{flux_type}.txt"
        elif method == "uniform":
            filename = f"{name}_spherex_spectrum_rebinned_uniform_{param:.3f}_{flux_type}.txt"
        else:
            filename = f"{name}_spherex_spectrum_{flux_type}.txt"
    else:
        filename = f"{name}_spherex_spectrum_{flux_type}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Create header with information
    header = f"# SPHEREx spectrum for {name}\n"
    if rebin_info is not None:
        method, param = rebin_info
        if method == "spherex":
            header += f"# Rebinned spectrum (SPHEREx method, tolerance = {param:.2f})\n"
        elif method == "uniform":
            header += f"# Rebinned spectrum (Uniform grid, Δλ = {param:.3f} μm)\n"
    header += f"# Flux type: {flux_type}\n"
    if flux_type == 'flam':
        header += "# Columns: wavelength(microns) flux(erg/s/cm²/Å) error(erg/s/cm²/Å)\n"
    elif flux_type == 'fnu':
        header += "# Columns: wavelength(microns) flux(mJy) error(mJy)\n"
    else:
        header += "# Columns: wavelength(microns) flux error\n"
    
    if rebin_info is not None:
        method, param = rebin_info
        if method == "spherex":
            header += "# Note: errors from proper error propagation (SPHEREx method)\n"
        elif method == "uniform":
            header += "# Note: errors are standard deviation of flux values in each wavelength bin\n"
    
    header += f"# Number of points: {len(wave)}\n"
    header += f"# Wavelength range: {wave.min():.3f} - {wave.max():.3f} microns\n"
    
    # Sort by wavelength
    sort_indices = np.argsort(wave)
    wave_sorted = wave[sort_indices]
    flux_sorted = flux[sort_indices]
    error_sorted = error[sort_indices]
    
    # Save data
    data = np.column_stack((wave_sorted, flux_sorted, error_sorted))
    np.savetxt(filepath, data, header=header[:-1], fmt='%.6e')  # Remove last newline from header
    
    print(f"Saved spectrum data: {filepath}")
    return filepath

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='''
        Extract SPHEREx photometry from cutout files and create plots.
        
        This script searches for SPHEREx cutout files matching the given object name,
        extracts photometry using aperture photometry on the central pixels,
        and creates publication-quality plots.
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES)

    parser.add_argument('--name', required=True, type=str,
                       help='Object name to search for in filenames')

    parser.add_argument('--directory', required=True, type=str,
                       help='Directory containing SPHEREx cutout files')

    parser.add_argument('--redshift', type=float, default=None,
                       help='Object redshift (for emission line markers and template redshifting)')

    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to files instead of displaying')

    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for saved plots (default: current directory)')

    parser.add_argument('--plot_flam', action='store_true', default=True,
                       help='Create f_lambda plot (default: True)')

    parser.add_argument('--plot_fnu', action='store_true',
                       help='Create f_nu plot (default: False unless specified)')

    parser.add_argument('--save_spectrum', action='store_true',
                       help='Save spectrum data to text files')

    parser.add_argument('--cr_thresh', type=float, default=10.0,
                       help='Cosmic ray threshold in MJy/sr (default: 10.0)')

    parser.add_argument('--show_template', type=str, default=None,
                       help='Show quasar template overlay (available: vandenberk01)')

    parser.add_argument('--show_rebin_spherex', type=float, default=None,
                       help='Show rebinned spectrum using SPHEREx-optimized method (specify tolerance, e.g., --show_rebin_spherex 0.1 for 10%% of bandwidth)')

    parser.add_argument('--show_rebin_uniform', type=float, default=None,
                       help='Show rebinned spectrum using uniform wavelength grid (specify delta wavelength in microns, e.g., --show_rebin_uniform 0.1)')

    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information during processing')

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Validate inputs
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        sys.exit(1)
    
    # Validate rebinning options
    if args.show_rebin_spherex is not None and args.show_rebin_uniform is not None:
        print("Error: Cannot specify both --show_rebin_spherex and --show_rebin_uniform")
        print("Please choose one rebinning method")
        sys.exit(1)
    
    # If neither plot type is explicitly specified, default to flam
    if not args.plot_fnu and not args.plot_flam:
        args.plot_flam = True
    
    # Find cutout files
    print(f"Searching for SPHEREx cutout files for '{args.name}' in '{args.directory}'")
    cutout_files = find_cutout_files(args.name, args.directory)
    
    if not cutout_files:
        print(f"No SPHEREx cutout files found for '{args.name}' in '{args.directory}'")
        print("\nFiles in directory:")
        all_files = glob.glob(os.path.join(args.directory, "*.fits"))
        for f in all_files[:10]:  # Show first 10 files as examples
            print(f"  {os.path.basename(f)}")
        if len(all_files) > 10:
            print(f"  ... and {len(all_files)-10} more files")
        sys.exit(1)
    
    print(f"Found {len(cutout_files)} cutout files")
    if args.verbose:
        for f in cutout_files:
            print(f"  {os.path.basename(f)}")
    else:
        # Show first few files as examples
        print("Example files:")
        for f in cutout_files[:3]:
            print(f"  {os.path.basename(f)}")
        if len(cutout_files) > 3:
            print(f"  ... and {len(cutout_files)-3} more files")
    
    # Load template if requested
    template_data = None
    if args.show_template is not None:
        if args.redshift is None:
            print("Warning: Template requested but no redshift provided. Using z=0.")
            redshift_for_template = 0.0
        else:
            redshift_for_template = args.redshift
            
        print(f"\nLoading template '{args.show_template}' at redshift z={redshift_for_template:.2f}")
        wave_template, flux_template = load_quasar_template(args.show_template, redshift_for_template)
        if wave_template is not None and flux_template is not None:
            template_data = (wave_template, flux_template)
        else:
            print("Failed to load template, continuing without template overlay.")
    
    # Extract photometry using our modified function that accepts a file list
    print("\nExtracting SPHEREx photometry...")
    
    # Add some diagnostic information about the first file
    if cutout_files:
        print(f"First file example: {os.path.basename(cutout_files[0])}")
        try:
            from astropy.io import fits
            with fits.open(cutout_files[0]) as hdul:
                print(f"File has {len(hdul)} extensions")
                if len(hdul) > 1:
                    img = hdul[1].data
                    print(f"Image shape: {img.shape}")
                    print(f"Image size (flattened): {len(img.flatten())}")
        except Exception as e:
            print(f"Could not examine first file: {e}")
    
    try:
        wave, dwave, flux, var = extract_spherex_from_files(cutout_files, cr_thresh=args.cr_thresh)
    except Exception as e:
        print(f"Error extracting photometry: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if len(wave) == 0:
        print("\nNo valid photometry points extracted!")
        print("\nPossible issues:")
        print("1. Cutout files might not be 6x6 pixels (36 pixels total)")
        print("2. Files might be corrupted or have wrong format")
        print("3. Cosmic ray threshold might be too low (try --cr_thresh 50)")
        print("4. Files might have different structure than expected")
        print("\nTry running with --verbose to see more details")
        sys.exit(1)
    
    print(f"Successfully extracted {len(wave)} photometry points")
    print(f"Wavelength range: {wave.min():.3f} - {wave.max():.3f} μm")
    
    # Convert to different flux units
    flam = get_flam(wave, flux)
    std_flam = get_flam(wave, np.sqrt(var))
    
    if args.plot_fnu:
        fnu = get_fnu_from_flam(wave, flam)
        std_fnu = get_fnu_from_flam(wave, std_flam)
    
    # Handle rebinning if requested
    rebin_method = None
    if args.show_rebin_spherex is not None:
        print(f"\nRebinning spectrum using SPHEREx method (tolerance = {args.show_rebin_spherex:.2f})...")
        try:
            # Use SPHEREx-specific rebinning
            wave_rebin_flam, flam_rebin, std_flam_rebin, dwave_rebin = rebin_spectrum_spherex(
                wave, flam, std_flam, dwave, tolerance=args.show_rebin_spherex)
            
            if args.plot_fnu:
                # Rebin f_nu data using SPHEREx method
                wave_rebin_fnu, fnu_rebin, std_fnu_rebin, dwave_rebin_fnu = rebin_spectrum_spherex(
                    wave, fnu, std_fnu, dwave, tolerance=args.show_rebin_spherex)
            
            # Use rebinned data for plotting and saving
            wave_plot_flam, flam_plot, std_flam_plot, dwave_plot_flam = wave_rebin_flam, flam_rebin, std_flam_rebin, dwave_rebin
            if args.plot_fnu:
                wave_plot_fnu, fnu_plot, std_fnu_plot, dwave_plot_fnu = wave_rebin_fnu, fnu_rebin, std_fnu_rebin, dwave_rebin_fnu
            
            rebin_method = "spherex"
            rebin_param = args.show_rebin_spherex
                
        except Exception as e:
            print(f"Error: SPHEREx rebinning failed: {e}")
            print("Please check that spherex_utils.py is available and working")
            sys.exit(1)
            
    elif args.show_rebin_uniform is not None:
        print(f"\nRebinning spectrum using uniform grid (Δλ = {args.show_rebin_uniform:.4f} μm)...")
        try:
            # Use uniform grid rebinning
            wave_rebin_flam, flam_rebin, std_flam_rebin, actual_delta_wave = rebin_spectrum(
                wave, flam, std_flam, args.show_rebin_uniform, 'flam')
            dwave_rebin_flam = np.full_like(wave_rebin_flam, actual_delta_wave)
            
            if args.plot_fnu:
                # Rebin f_nu data
                wave_rebin_fnu, fnu_rebin, std_fnu_rebin, _ = rebin_spectrum(
                    wave, fnu, std_fnu, args.show_rebin_uniform, 'fnu')
                dwave_rebin_fnu = np.full_like(wave_rebin_fnu, actual_delta_wave)
            
            # Use rebinned data for plotting and saving
            wave_plot_flam, flam_plot, std_flam_plot, dwave_plot_flam = wave_rebin_flam, flam_rebin, std_flam_rebin, dwave_rebin_flam
            if args.plot_fnu:
                wave_plot_fnu, fnu_plot, std_fnu_plot, dwave_plot_fnu = wave_rebin_fnu, fnu_rebin, std_fnu_rebin, dwave_rebin_fnu
            
            rebin_method = "uniform"
            rebin_param = args.show_rebin_uniform
                
        except Exception as e:
            print(f"Error: Uniform grid rebinning failed: {e}")
            sys.exit(1)
    else:
        # Use original data
        wave_plot_flam, flam_plot, std_flam_plot, dwave_plot_flam = wave, flam, std_flam, dwave
        if args.plot_fnu:
            wave_plot_fnu, fnu_plot, std_fnu_plot, dwave_plot_fnu = wave, fnu, std_fnu, dwave
        rebin_method = None
        rebin_param = None
    
    # Create output directory if saving plots
    if args.save_plots:
        create_output_directory(args.output_dir)
    
    # Create plots and save data
    print("\nCreating plots and saving data...")
    
    if args.plot_flam:
        print("Creating f_lambda plot...")
        if rebin_method is not None:
            # Use rebinned data for plotting
            plot_spherex_spectrum(wave_plot_flam, flam_plot, std_flam_plot, dwave_plot_flam, args.name, 
                                redshift=args.redshift, save_plots=args.save_plots,
                                output_dir=args.output_dir, plot_type='flam',
                                template_data=template_data, rebin_method=rebin_method, 
                                rebin_param=rebin_param)
        else:
            # Use original data for plotting
            plot_spherex_spectrum(wave, flam, std_flam, dwave, args.name, 
                                redshift=args.redshift, save_plots=args.save_plots,
                                output_dir=args.output_dir, plot_type='flam',
                                template_data=template_data, rebin_method=None, 
                                rebin_param=None)
        
        if args.save_spectrum:
            # Use appropriate suffix based on rebinning method
            if rebin_method == "spherex":
                save_spectrum_to_file(wave_plot_flam, flam_plot, std_flam_plot, args.name, 
                                    args.output_dir, flux_type='flam', rebin_info=(rebin_method, rebin_param))
            elif rebin_method == "uniform":
                save_spectrum_to_file(wave_plot_flam, flam_plot, std_flam_plot, args.name, 
                                    args.output_dir, flux_type='flam', rebin_info=(rebin_method, rebin_param))
            else:
                save_spectrum_to_file(wave_plot_flam, flam_plot, std_flam_plot, args.name, 
                                    args.output_dir, flux_type='flam', rebin_info=None)
    
    if args.plot_fnu:
        print("Creating f_nu plot...")
        if rebin_method is not None:
            # Use rebinned data for plotting
            plot_spherex_spectrum(wave_plot_fnu, fnu_plot, std_fnu_plot, dwave_plot_fnu, args.name,
                                redshift=args.redshift, save_plots=args.save_plots,
                                output_dir=args.output_dir, plot_type='fnu',
                                template_data=template_data, rebin_method=rebin_method,
                                rebin_param=rebin_param)
        else:
            # Use original data for plotting
            plot_spherex_spectrum(wave, fnu, std_fnu, dwave, args.name,
                                redshift=args.redshift, save_plots=args.save_plots,
                                output_dir=args.output_dir, plot_type='fnu',
                                template_data=template_data, rebin_method=None,
                                rebin_param=None)
        
        if args.save_spectrum:
            # Use appropriate suffix based on rebinning method
            if rebin_method == "spherex":
                save_spectrum_to_file(wave_plot_fnu, fnu_plot, std_fnu_plot, args.name, 
                                    args.output_dir, flux_type='fnu', rebin_info=(rebin_method, rebin_param))
            elif rebin_method == "uniform":
                save_spectrum_to_file(wave_plot_fnu, fnu_plot, std_fnu_plot, args.name, 
                                    args.output_dir, flux_type='fnu', rebin_info=(rebin_method, rebin_param))
            else:
                save_spectrum_to_file(wave_plot_fnu, fnu_plot, std_fnu_plot, args.name, 
                                    args.output_dir, flux_type='fnu', rebin_info=None)
    
    # Print summary statistics
    print(f"\nSummary for {args.name}:")
    if rebin_method is not None:
        if rebin_method == "spherex":
            print(f"Rebinning: SPHEREx method (tolerance = {rebin_param:.2f})")
        elif rebin_method == "uniform":
            print(f"Rebinning: Uniform grid (Δλ = {rebin_param:.4f} μm)")
        print(f"Number of spectral points (rebinned): {len(wave_plot_flam)}")
        print(f"Wavelength coverage: {wave_plot_flam.min():.3f} - {wave_plot_flam.max():.3f} μm")
        print(f"Median flux: {np.median(flam_plot):.2e} erg/s/cm²/Å")
        if np.any(std_flam_plot > 0):
            print(f"Median S/N: {np.median(flam_plot[std_flam_plot > 0]/std_flam_plot[std_flam_plot > 0]):.1f}")
        else:
            print("Median S/N: undefined (no error information)")
    else:
        print(f"Number of spectral points: {len(wave)}")
        print(f"Wavelength coverage: {wave.min():.3f} - {wave.max():.3f} μm")
        print(f"Median flux: {np.median(flam):.2e} erg/s/cm²/Å")
        print(f"Median S/N: {np.median(flam/std_flam):.1f}")
    
    if template_data is not None:
        print(f"Template plotted: {args.show_template} at z={args.redshift}")
    
    if args.save_plots or args.save_spectrum:
        print(f"\nFiles saved to: {args.output_dir}")
    
    if not args.save_plots:
        print("Plots displayed. Close plot windows to continue.")

if __name__ == '__main__':
    main()
