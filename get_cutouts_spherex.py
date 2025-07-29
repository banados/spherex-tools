#!/usr/bin/env python
"""
Eduardo Banados
Download SPHEREx data from IRSA for a list of coordinates
"""
from __future__ import print_function, division
import argparse
import os
import sys

import astropy.units as u
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
from astropy.io import ascii

try:
    from urllib2 import urlopen  # python2
    from httplib import IncompleteRead
    from urllib2 import HTTPError
except ImportError:
    from urllib.request import urlopen  # python3
    from urllib.error import HTTPError
    from http.client import IncompleteRead

EXAMPLES = """
Examples:

Create a text file (e.g., targets.txt) with columns: name ra dec
Example content:
NGC6888 308.027 38.35
M31 10.6847 41.2687
Crab 83.633 22.0145

Download full SPHEREx files:
python get_stamps_spherex.py -i targets.txt
python get_stamps_spherex.py -i targets.txt --radius 5

Download SPHEREx cutouts (much smaller files):
python get_stamps_spherex.py -i targets.txt --cutout
python get_stamps_spherex.py -i targets.txt --cutout --cutout_size 0.02
python get_stamps_spherex.py -i targets.txt --cutout --radius 10 --output_dir spherex_cutouts/

The script will query SPHEREx data within the specified radius and download
either full files or cutouts, naming them as:
- Full files: [name]_[original_filename].fits
- Cutouts: [name]_cutout_[original_filename].fits
"""

def read_input_file(filename, delimiter=None):
    """
    Read input file with name, ra, dec columns
    """
    try:
        # First, let's try reading with different methods
        print(f"Attempting to read file: {filename}")
        
        # Try reading with astropy's ascii.read
        try:
            if delimiter:
                data = ascii.read(filename, delimiter=delimiter)
            else:
                # Try common delimiters
                for delim in [None, ' ', '\t', ',']:
                    try:
                        data = ascii.read(filename, delimiter=delim)
                        print(f"Successfully read with delimiter: {repr(delim)}")
                        break
                    except:
                        continue
                else:
                    raise ValueError("Could not read file with any common delimiter")
                        
            print(f"Found columns: {data.colnames}")
            print(f"Number of rows: {len(data)}")
            
            # Convert column names to lowercase for easier matching
            data_cols_lower = [col.lower() for col in data.colnames]
            
            # Check if we have the required columns (case insensitive)
            required_cols = ['name', 'ra', 'dec']
            
            # Create mapping from actual column names to required names
            col_mapping = {}
            for req_col in required_cols:
                found = False
                for i, actual_col in enumerate(data_cols_lower):
                    if actual_col == req_col:
                        col_mapping[req_col] = data.colnames[i]
                        found = True
                        break
                
                if not found:
                    # Try alternative names
                    alternatives = {
                        'name': ['object', 'source', 'id', 'identifier', 'target'],
                        'ra': ['ra', 'right_ascension', 'alpha'],
                        'dec': ['dec', 'declination', 'delta']
                    }
                    
                    for alt_name in alternatives[req_col]:
                        for i, actual_col in enumerate(data_cols_lower):
                            if actual_col == alt_name.lower():
                                col_mapping[req_col] = data.colnames[i]
                                found = True
                                break
                        if found:
                            break
                
                if not found:
                    raise ValueError(f"Could not find column for '{req_col}'. Available columns: {data.colnames}")
            
            # Rename columns if necessary
            for req_col, actual_col in col_mapping.items():
                if actual_col != req_col:
                    data.rename_column(actual_col, req_col)
                    print(f"Renamed column '{actual_col}' to '{req_col}'")
            
            # Verify final columns
            if not all(col in data.colnames for col in required_cols):
                raise ValueError(f"Missing required columns. Found: {data.colnames}, Required: {required_cols}")
            
            print(f"Successfully processed columns: {required_cols}")
            return data
            
        except Exception as ascii_error:
            print(f"ASCII read failed: {ascii_error}")
            
            # Fallback: try reading as simple text file
            print("Trying manual file parsing...")
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                raise ValueError("File must have at least a header and one data row")
            
            # Parse header
            header = lines[0].strip().split()
            print(f"Header from manual parsing: {header}")
            
            # Check if header contains required columns (case insensitive)
            header_lower = [h.lower() for h in header]
            required_indices = {}
            
            for req_col in required_cols:
                try:
                    idx = header_lower.index(req_col)
                    required_indices[req_col] = idx
                except ValueError:
                    raise ValueError(f"Column '{req_col}' not found in header: {header}")
            
            # Parse data rows
            data_rows = []
            for line in lines[1:]:
                if line.strip():  # Skip empty lines
                    parts = line.strip().split()
                    if len(parts) >= len(header):
                        row_data = {}
                        for req_col, idx in required_indices.items():
                            row_data[req_col] = parts[idx]
                        data_rows.append(row_data)
            
            # Convert to astropy table
            from astropy.table import Table
            
            if not data_rows:
                raise ValueError("No data rows found")
            
            # Create table
            table_data = {col: [row[col] for row in data_rows] for col in required_cols}
            
            # Convert ra and dec to float
            try:
                table_data['ra'] = [float(x) for x in table_data['ra']]
                table_data['dec'] = [float(x) for x in table_data['dec']]
            except ValueError as e:
                raise ValueError(f"Could not convert coordinates to float: {e}")
            
            data = Table(table_data)
            print(f"Manual parsing successful. Rows: {len(data)}")
            return data
            
    except Exception as e:
        print(f"Error reading input file: {e}")
        print("\nFile format should be:")
        print("name ra dec")
        print("object1 123.456 78.901")
        print("object2 234.567 -12.345")
        print("\nMake sure:")
        print("- File has a header row with column names")
        print("- Columns are separated by spaces, tabs, or commas")
        print("- RA and Dec are in decimal degrees")
        sys.exit(1)

def query_spherex_data(ra, dec, radius_arcsec=3):
    """
    Query SPHEREx data at given coordinates within radius
    
    Parameters:
    -----------
    ra : float
        Right ascension in degrees
    dec : float  
        Declination in degrees
    radius_arcsec : float
        Search radius in arcseconds
        
    Returns:
    --------
    access_urls : list
        List of access URLs for SPHEREx data
    """
    try:
        # Create coordinate object
        coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        
        # Query SPHEREx data
        print(f"Querying SPHEREx data at RA={ra:.6f}, Dec={dec:.6f} within {radius_arcsec}\"")
        rslt = Irsa.query_sia(pos=(coord.ra, coord.dec, radius_arcsec*u.arcsec), 
                             instrument="SPHEREx")
        
        if len(rslt) == 0:
            print("No SPHEREx data found at this position")
            return []
            
        print(f"Found {len(rslt)} SPHEREx file(s)")
        
        # Get access URLs
        access_urls = rslt['access_url']
        return list(access_urls)
        
    except Exception as e:
        print(f"Error querying SPHEREx data: {e}")
        return []

def build_download_url(access_url, cutout=False, ra=None, dec=None, cutout_size=0.01):
    """
    Build full download URL by adding IRSA prefix
    For cutouts, converts the URL to use the cutout service
    
    Parameters:
    -----------
    access_url : str
        Original access URL from IRSA query
    cutout : bool
        If True, build a cutout URL instead of full file
    ra : float
        Right ascension for cutout center (degrees)
    dec : float  
        Declination for cutout center (degrees)
    cutout_size : float
        Cutout size in degrees
    """
    base_url = "https://irsa.ipac.caltech.edu/"
    
    if cutout and ra is not None and dec is not None:
        # Convert to cutout URL
        # Original: /ibe/data/spherex/qr/level2/...
        # Cutout: /ibe/cutout?ra={ra}&dec={dec}&size={size}&path=spherex/qr/level2/...
        
        if access_url.startswith('/'):
            access_url = access_url[1:]  # Remove leading slash
        
        # Extract the path after /ibe/data/
        if access_url.startswith('ibe/data/'):
            data_path = access_url[9:]  # Remove 'ibe/data/'
        else:
            # Handle case where access_url might not start with ibe/data/
            data_path = access_url
        
        cutout_url = f"{base_url}ibe/cutout?ra={ra}&dec={dec}&size={cutout_size}&path={data_path}"
        return cutout_url
    else:
        # Standard full file download
        if access_url.startswith('/'):
            return base_url + access_url[1:]  # Remove leading slash if present
        else:
            return base_url + access_url

def extract_filename_from_url(url):
    """
    Extract the original filename from the download URL
    """
    return url.split('/')[-1]

def download_file(url, output_filename):
    """
    Download file from URL and save with given filename
    
    Returns:
    --------
    success : bool
        True if download was successful
    """
    try:
        print(f"Downloading from: {url}")
        response = urlopen(url)
        
        # Always download the file data first
        print("Reading file data...")
        file_data = response.read()
        actual_size = len(file_data)
        
        print(f"Downloaded {actual_size:,} bytes ({actual_size/1024/1024:.2f} MB)")
        
        # Only reject if extremely small (< 100 bytes, likely an error page)
        if actual_size < 100:
            print(f"Warning: File is very small ({actual_size} bytes)")
            try:
                content_str = file_data.decode('utf-8', errors='ignore')
                if any(keyword in content_str.lower() for keyword in ['error', 'not found', '404', 'forbidden']):
                    print(f"File appears to be an error page: {content_str[:200]}")
                    return False
            except:
                pass
        
        # Save the file
        with open(output_filename, 'wb') as output_file:
            output_file.write(file_data)
            
        # Verify the file was saved correctly
        if os.path.exists(output_filename):
            saved_size = os.path.getsize(output_filename)
            print(f"Successfully saved {output_filename} ({saved_size:,} bytes)")
            if saved_size != actual_size:
                print(f"Warning: Size mismatch! Downloaded: {actual_size}, Saved: {saved_size}")
            return True
        else:
            print(f"Error: Failed to create file {output_filename}")
            return False
            
    except (IncompleteRead, HTTPError) as err:
        print(f"Network error downloading {url}: {err}")
        return False
    except Exception as e:
        print(f"Unexpected error downloading {url}: {e}")
        import traceback
        traceback.print_exc()
        return False

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='''
        Download SPHEREx data from IRSA for a list of coordinates.
        
        Input file should contain columns: name ra dec
        where ra and dec are in decimal degrees.
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES)

    parser.add_argument('-i', '--input', required=True, type=str,
                       help='Input file containing name, ra, dec columns')

    parser.add_argument('--radius', required=False, type=float, default=3.0,
                       help='Search radius in arcseconds (default: 3.0)')

    parser.add_argument('--output_dir', required=False, type=str, default='.',
                       help='Output directory for downloaded files (default: current directory)')

    parser.add_argument('--delimiter', required=False, type=str, default=None,
                       help='Column delimiter in input file (default: auto-detect)')

    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing files (default: skip existing files)')

    parser.add_argument('--cutout', action='store_true',
                       help='Download cutouts instead of full files')

    parser.add_argument('--cutout_size', required=False, type=float, default=0.01,
                       help='Cutout size in degrees (default: 0.01 = 36 arcsec)')

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    # Read input file
    print(f"Reading input file: {args.input}")
    data = read_input_file(args.input, delimiter=args.delimiter)
    print(f"Found {len(data)} targets")
    
    # Display mode information
    if args.cutout:
        cutout_arcsec = args.cutout_size * 3600  # Convert degrees to arcseconds
        print(f"Mode: Downloading cutouts (size: {args.cutout_size}° = {cutout_arcsec:.1f}\")")
    else:
        print("Mode: Downloading full files")
    print(f"Search radius: {args.radius}\"")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Statistics
    total_targets = len(data)
    targets_with_data = 0
    total_files_downloaded = 0
    targets_without_data = 0
    
    # Process each target
    for i, row in enumerate(data):
        name = str(row['name']).strip()
        ra = float(row['ra'])
        dec = float(row['dec'])
        
        print(f"\n{'='*60}")
        print(f"Processing target {i+1}/{total_targets}: {name}")
        print(f"Coordinates: RA={ra:.6f}°, Dec={dec:.6f}°")
        
        # Query SPHEREx data
        access_urls = query_spherex_data(ra, dec, args.radius)
        
        if not access_urls:
            print(f"No SPHEREx data found for {name}")
            targets_without_data += 1
            continue
            
        targets_with_data += 1
        files_downloaded_for_target = 0
        
        # Download each file
        for j, access_url in enumerate(access_urls):
            # Build full download URL
            download_url = build_download_url(access_url, 
                                            cutout=args.cutout,
                                            ra=ra, 
                                            dec=dec, 
                                            cutout_size=args.cutout_size)
            
            # Extract original filename
            original_filename = extract_filename_from_url(access_url)  # Use original access_url for filename
            
            # Create output filename
            if args.cutout:
                # Insert "cutout" before the file extension
                name_part, ext = os.path.splitext(original_filename)
                output_filename = f"{name}_cutout_{name_part}{ext}"
            else:
                output_filename = f"{name}_{original_filename}"
                
            output_path = os.path.join(args.output_dir, output_filename)
            
            # Check if file already exists
            if os.path.exists(output_path) and not args.overwrite:
                print(f"File already exists, skipping: {output_filename}")
                continue
            
            # Download file
            if download_file(download_url, output_path):
                files_downloaded_for_target += 1
                total_files_downloaded += 1
        
        print(f"Downloaded {files_downloaded_for_target} file(s) for {name}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Total targets processed: {total_targets}")
    print(f"Targets with SPHEREx data: {targets_with_data}")
    print(f"Targets without SPHEREx data: {targets_without_data}")
    print(f"Total files downloaded: {total_files_downloaded}")
    print(f"Output directory: {args.output_dir}")

if __name__ == '__main__':
    main()