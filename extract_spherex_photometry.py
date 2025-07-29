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
            img = hdul[1].data
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
                wx = hdul[2].data['X'][0]
                wy = hdul[2].data['Y'][0]
                wval = hdul[2].data['VALUES'][0][:, :, 0]
                dwval = hdul[2].data['VALUES'][0][:, :, 1]
                interp_wave = RegularGridInterpolator((wx, wy), wval)
                interp_dwave = RegularGridInterpolator((wx, wy), dwval)
                wave[ii] = interp_wave((3 - hdul[1].header['CRPIX2W'], 3 - hdul[1].header['CRPIX1W']))
                dwave[ii] = interp_dwave((3 - hdul[1].header['CRPIX2W'], 3 - hdul[1].header['CRPIX1W']))
                
                # Extremely rough estimate of the variance (full images have a proper variance map, but not cutouts...)
                # First remove the highest and lowest pixels from the sky (sort of like outlier masking)
                skymask = sky & (img > img[sky].min()) & (img < img[sky].max())
                # Variance of each pixel is first approximated by variance of sky pixels
                var[ii] = np.sum(mask) * np.var(img[skymask])
                # Now we adjust the variance higher assuming that variance is proportional to flux
                # but we only include the pixels above the sky background to avoid decreasing it
                posmask = img[mask] > np.median(img[sky])
                var[ii] *= 1 + np.sum(((img[mask] - np.median(img[sky])) / np.median(img[sky]))[posmask])
                
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

EXAMPLES = """
Examples:

# Extract photometry for an object with cutouts in a specific directory
python extract_spherex_photometry.py --name NGC6888 --directory cutouts/

# Specify redshift for emission line markers
python extract_spherex_photometry.py --name J0916-2511 --directory data/ --redshift 4.85

# Save plots and spectrum data to files
python extract_spherex_photometry.py --name M31 --directory spherex_data/ --save_plots --save_spectrum

# Custom output directory for plots and data
python extract_spherex_photometry.py --name Crab --directory data/ --save_plots --save_spectrum --output_dir results/

# Both flux density units with saved data
python extract_spherex_photometry.py --name NGC2024 --directory cutouts/ --plot_fnu --plot_flam --save_spectrum

Expected directory structure:
cutouts/
├── NGC6888_cutout_level2_2025W19_1B_0501_2D5_spx_l2b-v11-2025-162.fits
├── NGC6888_cutout_level2_2025W19_1B_0501_2D2_spx_l2b-v11-2025-162.fits
└── ...

Output files:
├── NGC6888_spherex_flam.png/pdf (plots)
├── NGC6888_spherex_fnu.png/pdf (plots)  
├── NGC6888_spherex_spectrum_flam.txt (spectral data)
└── NGC6888_spherex_spectrum_fnu.txt (spectral data)
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
                         save_plots=False, output_dir='.', plot_type='flam'):
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
    """
    
    # Create label
    if redshift is not None:
        label = f"{name}, z={redshift:.2f}"
    else:
        label = name
    
    # Create the plot but don't show it yet if we're saving
    import matplotlib.pyplot as plt
    
    if plot_type == 'flam':
        # Create flam plot manually to control saving
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        plt.errorbar(wave, flam, c='k', fmt='o', xerr=0.5*dwave, yerr=std, label=label, ms=3.5)
        plt.axhline(0.0, c='darkorange', lw=0.5, linestyle='dashed')
        
        if label:
            plt.legend(fontsize=16)
        
        plt.ylim(-0.2*flam.max(), 1.2*flam.max())
        plt.xlim(0.65, 5.05)
        
        if redshift is not None:
            line_list = [6564.6, 5008.2, 4862.7, 2800.0, 1908.7, 1550.0, 1215.67]
            for line in line_list:
                line_wave = line * (1 + redshift) * 1e-4
                if 0.65 <= line_wave <= 5.05:
                    plt.axvline(line_wave, c='k', linestyle='dotted', alpha=0.7)
        
        plt.ylabel('F$_λ$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]', fontsize=18)
        plt.xlabel('Wavelength [μm]', fontsize=18)
        plt.tick_params(which='both', top=True, right=True, direction='in', labelsize=13)
        plt.minorticks_on()
        plt.tight_layout()
        
    elif plot_type == 'fnu':
        # Create fnu plot manually to control saving
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        plt.errorbar(wave, flam, c='k', fmt='o', xerr=0.5*dwave, yerr=std, label=label, ms=3.5)
        plt.axhline(0.0, c='darkorange', lw=0.5, linestyle='dashed')
        
        if label:
            plt.legend(fontsize=16, loc='upper left')
        
        plt.ylim(-0.2*flam.max(), 1.2*flam.max())
        plt.xlim(0.65, 5.05)
        
        if redshift is not None:
            line_list = [6564.6, 5008.2, 4862.7, 2800.0, 1908.7, 1550.0, 1215.67]
            for line in line_list:
                line_wave = line * (1 + redshift) * 1e-4
                if 0.65 <= line_wave <= 5.05:
                    plt.axvline(line_wave, c='k', linestyle='dotted', alpha=0.7)
        
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

def save_spectrum_to_file(wave, flux, error, name, output_dir='.', flux_type='flam'):
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
    """
    filename = f"{name}_spherex_spectrum_{flux_type}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Create header with information
    header = f"# SPHEREx spectrum for {name}\n"
    header += f"# Flux type: {flux_type}\n"
    if flux_type == 'flam':
        header += "# Columns: wavelength(microns) flux(erg/s/cm²/Å) error(erg/s/cm²/Å)\n"
    elif flux_type == 'fnu':
        header += "# Columns: wavelength(microns) flux(mJy) error(mJy)\n"
    else:
        header += "# Columns: wavelength(microns) flux error\n"
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
                       help='Object redshift (for emission line markers)')

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

    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information during processing')

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Validate inputs
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
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
    
    # Create output directory if saving plots
    if args.save_plots:
        create_output_directory(args.output_dir)
    
    # Create plots and save data
    print("\nCreating plots and saving data...")
    
    if args.plot_flam:
        print("Creating f_lambda plot...")
        plot_spherex_spectrum(wave, flam, std_flam, dwave, args.name, 
                            redshift=args.redshift, save_plots=args.save_plots,
                            output_dir=args.output_dir, plot_type='flam')
        
        if args.save_spectrum:
            save_spectrum_to_file(wave, flam, std_flam, args.name, 
                                args.output_dir, flux_type='flam')
    
    if args.plot_fnu:
        print("Creating f_nu plot...")
        plot_spherex_spectrum(wave, fnu, std_fnu, dwave, args.name,
                            redshift=args.redshift, save_plots=args.save_plots,
                            output_dir=args.output_dir, plot_type='fnu')
        
        if args.save_spectrum:
            save_spectrum_to_file(wave, fnu, std_fnu, args.name, 
                                args.output_dir, flux_type='fnu')
    
    # Print summary statistics
    print(f"\nSummary for {args.name}:")
    print(f"Number of spectral points: {len(wave)}")
    print(f"Wavelength coverage: {wave.min():.3f} - {wave.max():.3f} μm")
    print(f"Median flux: {np.median(flam):.2e} erg/s/cm²/Å")
    print(f"Median S/N: {np.median(flam/std_flam):.1f}")
    
    if args.save_plots or args.save_spectrum:
        print(f"\nFiles saved to: {args.output_dir}")
    
    if not args.save_plots:
        print("\nPlots displayed. Close plot windows to continue.")

if __name__ == '__main__':
    main()