# SPHEREx Analysis Toolkit

A complete Python toolkit for downloading and analyzing SPHEREx (Spectro-Photometer for the History of the Universe, Epoch of Reionization and Ices Explorer) data from IRSA.

## Features

### 📥 Data Download (`get_cutouts_spherex.py`)
- **Automated SPHEREx data queries** using astroquery
- **Full file downloads** (complete SPHEREx FITS files, ~80MB each)
- **Cutout downloads** (small stamps centered on your coordinates, much faster)
- **Batch processing** from simple text files
- **Smart file naming** with source names
- **Resume capability** (skips already downloaded files)

### 📊 Photometry Extraction (`extract_spherex_photometry.py`)
- **Automated aperture photometry** from SPHEREx cutouts
- **Publication-quality plots** in both f_λ and f_ν units
- **Spectral data export** to text files
- **Emission line markers** for redshifted sources
- **Flexible file search** across different naming conventions
- **Comprehensive error handling** and diagnostics

*The photometry extraction components are based on [spherex_quicklook](https://github.com/freddavies/spherex_quicklook) by Fred Davies.*

### 🔧 Analysis Library (`spherex_utils.py`)
- **Core extraction functions** for SPHEREx data processing  
- **Unit conversions** between f_λ and f_ν
- **Plotting utilities** with customizable styling
- **Spectral rebinning** and analysis tools
- **Quality assessment** and statistics

## Installation

### Option 1: Using Conda (Recommended)

Create a dedicated conda environment for SPHEREx analysis:

```bash
# Create environment
conda create -n spherex-analysis python=3.11 -c conda-forge
conda activate spherex-analysis

# Install dependencies
conda install -c conda-forge astropy matplotlib jupyter scipy
pip install astroquery>=0.4.10
```

### Option 2: Using pip

```bash
pip install -r requirements.txt
```

## Complete Workflow

### Step 1: Download SPHEREx Data

Create a text file with three columns: `name ra dec`

```
name ra dec
NGC6888 308.027 38.35
M31 10.6847 41.2687
Crab 83.633 22.0145
```

Download cutouts (recommended for spectroscopy):
```bash
python get_cutouts_spherex.py -i targets.txt --cutout --output_dir cutouts/
```

### Step 2: Extract Photometry and Create Plots

```bash
python extract_spherex_photometry.py --name NGC6888 --directory cutouts/ --save_plots --save_spectrum --output_dir results/
```

## Detailed Usage

### 📥 Data Download Options

```bash
# Download full SPHEREx files
python get_cutouts_spherex.py -i targets.txt

# Download small cutouts (recommended for quick analysis)
python get_cutouts_spherex.py -i targets.txt --cutout

# Custom search radius and cutout size
python get_cutouts_spherex.py -i targets.txt --cutout --radius 5 --cutout_size 0.02

# Specify output directory
python get_cutouts_spherex.py -i targets.txt --cutout --output_dir spherex_data/

# Overwrite existing files
python get_cutouts_spherex.py -i targets.txt --overwrite

# Combine options
python get_cutouts_spherex.py -i targets.txt --cutout --radius 10 --cutout_size 0.015 --output_dir cutouts/
```

### 📊 Photometry Extraction Options

```bash
# Basic extraction with plots
python extract_spherex_photometry.py --name NGC6888 --directory cutouts/

# With redshift for emission line markers
python extract_spherex_photometry.py --name J0916-2511 --directory data/ --redshift 4.85

# Save plots and spectral data
python extract_spherex_photometry.py --name M31 --directory cutouts/ --save_plots --save_spectrum

# Both f_lambda and f_nu analysis
python extract_spherex_photometry.py --name Crab --directory cutouts/ --plot_flam --plot_fnu --save_spectrum

# Custom output directory
python extract_spherex_photometry.py --name NGC2024 --directory cutouts/ --save_plots --save_spectrum --output_dir results/
```

## Command Line Options

### Download Script (`get_cutouts_spherex.py`)

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input file with coordinates | Required |
| `--radius` | Search radius in arcseconds | 3.0 |
| `--cutout` | Download cutouts instead of full files | False |
| `--cutout_size` | Cutout size in degrees | 0.01 (36") |
| `--output_dir` | Output directory | Current directory |
| `--overwrite` | Overwrite existing files | False |
| `--delimiter` | Column delimiter in input file | Auto-detect |

### Photometry Script (`extract_spherex_photometry.py`)

| Option | Description | Default |
|--------|-------------|---------|
| `--name` | Object name to search for | Required |
| `--directory` | Directory with cutout files | Required |
| `--redshift` | Object redshift (for line markers) | None |
| `--save_plots` | Save plots to files | False |
| `--save_spectrum` | Save spectral data to text files | False |
| `--output_dir` | Output directory | Current directory |
| `--plot_flam` | Create f_λ plot | True |
| `--plot_fnu` | Create f_ν plot | False |
| `--cr_thresh` | Cosmic ray threshold (MJy/sr) | 10.0 |
| `--verbose` | Print detailed processing info | False |

## Output Files

### Downloaded Data
- **Full files**: `{name}_{original_spherex_filename}.fits` (~80MB each)
- **Cutouts**: `{name}_cutout_{original_spherex_filename}.fits` (~few KB each)

### Analysis Products
- **Plots**: `{name}_spherex_{flam|fnu}.{png|pdf}`
- **Spectra**: `{name}_spherex_spectrum_{flam|fnu}.txt`

### Spectral Data Format
```
# SPHEREx spectrum for NGC6888
# Flux type: flam  
# Columns: wavelength(microns) flux(erg/s/cm²/Å) error(erg/s/cm²/Å)
# Number of points: 2736
# Wavelength range: 0.750 - 4.820 microns
7.502340e-01  1.234567e-16  2.345678e-17
7.514560e-01  1.345678e-16  2.456789e-17
...
```

## Examples

### 🚀 Quick Start
```bash
# Download example cutouts
python get_cutouts_spherex.py -i example_targets.txt --cutout --output_dir cutouts/

# Extract photometry for first object
python extract_spherex_photometry.py --name NGC6888 --directory cutouts/ --save_spectrum
```

### 🔬 Complete Analysis Pipeline
```bash
# 1. Download cutouts for a survey
python get_cutouts_spherex.py -i survey_targets.txt --cutout --output_dir survey_cutouts/

# 2. Process each object with redshift information
python extract_spherex_photometry.py --name QSO_J0916 --directory survey_cutouts/ --redshift 4.85 --save_plots --save_spectrum --output_dir results/

# 3. Create publication plots
python extract_spherex_photometry.py --name QSO_J0916 --directory survey_cutouts/ --redshift 4.85 --plot_flam --plot_fnu --save_plots --output_dir paper_plots/
```

### 📈 Large Survey Mode
```bash
# Download full files for detailed analysis (warning: large files!)
python get_cutouts_spherex.py -i survey_targets.txt --radius 5 --output_dir full_survey/
```

## SPHEREx Mission Information

SPHEREx launched in March 2025 and provides:
- **All-sky near-infrared spectroscopy** (0.75-5.0 μm)
- **96 spectral channels** across the wavelength range  
- **Regular data releases** through IRSA
- **Unprecedented spectral survey** of the entire sky

## File Structure

After running the complete workflow, your directory structure will look like:

```
project/
├── get_cutouts_spherex.py         # Data download script
├── extract_spherex_photometry.py  # Photometry extraction script  
├── spherex_utils.py               # Analysis library
├── targets.txt                    # Input coordinate file
├── cutouts/                       # Downloaded cutout files
│   ├── NGC6888_cutout_level2_*.fits
│   └── M31_cutout_level2_*.fits
└── results/                       # Analysis products
    ├── NGC6888_spherex_flam.png
    ├── NGC6888_spherex_flam.pdf
    ├── NGC6888_spherex_spectrum_flam.txt
    └── ...
```

## Troubleshooting

### Common Issues

**"No SPHEREx data found"**
- Try increasing the search radius: `--radius 10`
- Check your coordinates are in decimal degrees
- Verify the object is in SPHEREx coverage area

**"No valid photometry points extracted"**
- Check that cutout files are properly formatted (6×6 pixels)
- Try increasing cosmic ray threshold: `--cr_thresh 50`
- Use `--verbose` flag for detailed diagnostics

**"File already exists"**
- Use `--overwrite` to replace existing files
- Or move/rename existing files

**Import errors**
- Ensure you have astroquery ≥ 0.4.10: `pip install --upgrade astroquery`
- Check your conda environment is activated

### Getting Help

```bash
python get_cutouts_spherex.py --help
python extract_spherex_photometry.py --help
```

## Dependencies

- Python ≥ 3.9
- astropy ≥ 5.0
- astroquery ≥ 0.4.10
- matplotlib ≥ 3.5
- scipy ≥ 1.9
- numpy ≥ 1.20

## Contributing

Issues and pull requests welcome! Please ensure any changes maintain compatibility with the current IRSA SPHEREx archive structure.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:
- The SPHEREx mission papers
- IRSA/IPAC for data hosting
- astroquery for the query interface
- The original [spherex_quicklook](https://github.com/freddavies/spherex_quicklook) repository for the photometry extraction methods

## Acknowledgments

- **SPHEREx Team** for the amazing mission and data
- **IRSA/IPAC** for data hosting and access infrastructure  
- **astroquery developers** for the query interface
- **astropy community** for the foundational tools
- **Fred Davies** for the original [spherex_quicklook](https://github.com/freddavies/spherex_quicklook) photometry extraction code