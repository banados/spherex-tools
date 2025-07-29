# SPHEREx Data Downloader

A Python script to download SPHEREx data from IRSA for a list of astronomical coordinates.

## Features

- **Automated SPHEREx data queries** using astroquery
- **Full file downloads** (complete SPHEREx FITS files, ~80MB each)
- **Cutout downloads** (small stamps centered on your coordinates, much faster)
- **Batch processing** from simple text files
- **Smart file naming** with source names
- **Resume capability** (skips already downloaded files)

## Installation

### Option 1: Using Conda (Recommended)

Create a dedicated conda environment for SPHEREx analysis:

```bash
# Create environment
conda create -n spherex-analysis python=3.11 -c conda-forge
conda activate spherex-analysis

# Install dependencies
conda install -c conda-forge astropy matplotlib jupyter
pip install astroquery>=0.4.10
```

### Option 2: Using pip

```bash
pip install -r requirements.txt
```

## Usage

### Input File Format

Create a text file with three columns: `name ra dec`

```
name ra dec
NGC6888 308.027 38.35
M31 10.6847 41.2687
Crab 83.633 22.0145
```

- **name**: Object identifier (used in output filenames)
- **ra**: Right ascension in decimal degrees
- **dec**: Declination in decimal degrees

### Basic Usage

```bash
# Download full SPHEREx files
python get_cutouts_spherex.py -i targets.txt

# Download small cutouts (recommended for quick analysis)
python get_cutouts_spherex.py -i targets.txt --cutout
```

### Advanced Options

```bash
# Custom search radius
python get_cutouts_spherex.py -i targets.txt --radius 5

# Custom cutout size (in degrees)
python get_cutouts_spherex.py -i targets.txt --cutout --cutout_size 0.02

# Specify output directory
python get_cutouts_spherex.py -i targets.txt --output_dir spherex_data/

# Overwrite existing files
python get_cutouts_spherex.py -i targets.txt --overwrite

# Combine options
python get_cutouts_spherex.py -i targets.txt --cutout --radius 10 --cutout_size 0.015 --output_dir cutouts/
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input file with coordinates | Required |
| `--radius` | Search radius in arcseconds | 3.0 |
| `--cutout` | Download cutouts instead of full files | False |
| `--cutout_size` | Cutout size in degrees | 0.01 (36") |
| `--output_dir` | Output directory | Current directory |
| `--overwrite` | Overwrite existing files | False |
| `--delimiter` | Column delimiter in input file | Auto-detect |

## Output Files

### Full Files
- **Filename**: `{name}_{original_spherex_filename}.fits`
- **Example**: `NGC6888_level2_2025W19_1B_0501_2D5_spx_l2b-v11-2025-162.fits`
- **Size**: ~80MB per file
- **Content**: Complete SPHEREx spectral data

### Cutouts
- **Filename**: `{name}_cutout_{original_spherex_filename}.fits`  
- **Example**: `NGC6888_cutout_level2_2025W19_1B_0501_2D5_spx_l2b-v11-2025-162.fits`
- **Size**: Much smaller (~few KB)
- **Content**: Small stamp centered on your coordinates

## Examples

### Quick Start
```bash
# Test with the provided example file
python get_cutouts_spherex.py -i example_targets.txt --cutout

# Download cutouts for multiple objects
python get_cutouts_spherex.py -i my_galaxies.txt --cutout --output_dir galaxy_cutouts/
```

### Large Survey Mode
```bash
# Download full files for detailed analysis (warning: large files!)
python get_cutouts_spherex.py -i survey_targets.txt --radius 5 --output_dir full_survey/
```


## Troubleshooting

### Common Issues


**"File already exists"**
- Use `--overwrite` to replace existing files
- Or move/rename existing files

**Import errors**
- Ensure you have astroquery ≥ 0.4.10: `pip install --upgrade astroquery`
- Check your conda environment is activated

### Getting Help

```bash
python get_cutouts_spherex.py --help
```

## Dependencies

- Python ≥ 3.9
- astropy ≥ 5.0
- astroquery ≥ 0.4.10

## Contributing

Issues and pull requests welcome! Please ensure any changes maintain compatibility with the current IRSA SPHEREx archive structure.

## License

MIT License - see LICENSE file for details.

