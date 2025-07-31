#!/usr/bin/env python
"""
Check SPHEREx data availability for a list of coordinates
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import astropy.units as u
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.table import Table, Column

EXAMPLES = """
Examples:

# Check SPHEREx coverage for targets in a file
python check_spherex_coverage.py -i targets.txt

# Works with CSV files (preserves all columns)
python check_spherex_coverage.py -i catalog.csv

# Use custom search radius  
python check_spherex_coverage.py -i targets.txt --radius 5

# Save to different output file
python check_spherex_coverage.py -i targets.txt -o targets_with_spherex.txt

# Update existing file with spherex_data column
python check_spherex_coverage.py -i my_catalog.csv --update

# Check with progress reporting every 10 objects
python check_spherex_coverage.py -i large_catalog.txt --progress 10

# Retry only the sources that had query errors (spherex_data = -1)
python check_spherex_coverage.py -i catalog.csv --retry-errors

# Retry errors with frequent progress updates
python check_spherex_coverage.py -i catalog.csv --retry-errors --progress 5

Input file formats supported:
- CSV: catalog.csv with any number of columns
- Space-separated: targets.txt  
- Tab-separated: data.tsv
- Must contain columns: name, ra, dec (case insensitive)
- All other columns are preserved

Example input (minimal):
name,ra,dec
NGC6888,308.027,38.35
M31,10.6847,41.2687

Example input (with extra columns):
object_id,name,ra,dec,magnitude,classification,notes
1,NGC6888,308.027,38.35,12.1,nebula,bright
2,M31,10.6847,41.2687,3.4,galaxy,andromeda

After first run (some errors):
object_id,name,ra,dec,magnitude,classification,notes,spherex_data
1,NGC6888,308.027,38.35,12.1,nebula,bright,147
2,M31,10.6847,41.2687,3.4,galaxy,andromeda,-1

After retry-errors:
object_id,name,ra,dec,magnitude,classification,notes,spherex_data
1,NGC6888,308.027,38.35,12.1,nebula,bright,147
2,M31,10.6847,41.2687,3.4,galaxy,andromeda,0

The spherex_data column contains:
- -1: Query error (retry with --retry-errors)
- 0: No SPHEREx data found  
- N: Number of SPHEREx images available at that position
"""

def read_input_file(filename, delimiter=None, name_column=None):
    """
    Read input file with name, ra, dec columns (can have additional columns)
    Supports CSV, text, and FITS table formats
    """
    try:
        print(f"Reading file: {filename}")
        
        # Check if it's a FITS file
        _, ext = os.path.splitext(filename.lower())
        
        if ext in ['.fits', '.fit']:
            # Handle FITS file
            print("Detected FITS file format")
            from astropy.io import fits
            from astropy.table import Table
            
            # Open FITS file and look for table data
            with fits.open(filename) as hdul:
                print(f"FITS file has {len(hdul)} extensions")
                
                # Look for table data in extensions
                table_data = None
                for i, hdu in enumerate(hdul):
                    if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                        print(f"Found table data in extension {i}")
                        table_data = Table.read(filename, hdu=i)
                        break
                
                if table_data is None:
                    # Try reading as primary HDU
                    try:
                        table_data = Table.read(filename)
                        print("Read table from primary HDU")
                    except:
                        raise ValueError("No table data found in FITS file")
                
                data = table_data
                print(f"Successfully read FITS table")
        
        else:
            # Handle text/CSV files (existing code)
            data = None
            
            # First try: Let astropy auto-detect everything
            try:
                data = ascii.read(filename, delimiter=delimiter)
                print(f"Successfully read with astropy auto-detection")
            except:
                pass
            
            # Second try: Common delimiters for CSV and space-separated files
            if data is None:
                for delim in [',', '\t', ' ', '|', ';']:
                    try:
                        data = ascii.read(filename, delimiter=delim)
                        print(f"Successfully read with delimiter: '{delim}'")
                        break
                    except:
                        continue
            
            # Third try: Use pandas as fallback (handles CSV better)
            if data is None:
                try:
                    import pandas as pd
                    df = pd.read_csv(filename)
                    from astropy.table import Table
                    data = Table.from_pandas(df)
                    print(f"Successfully read with pandas")
                except ImportError:
                    print("Note: pandas not available for CSV fallback")
                except:
                    pass
            
            # Final fallback: Try reading as space-separated ignoring bad lines
            if data is None:
                try:
                    data = ascii.read(filename, delimiter=None, header_start=0, data_start=1)
                    print(f"Successfully read with fallback method")
                except:
                    pass
            
            if data is None:
                raise ValueError("Could not read file with any method")
                        
        print(f"Read {len(data)} rows with {len(data.colnames)} columns")
        print(f"Available columns: {data.colnames}")
        
        # Convert column names to lowercase for matching
        data_cols_lower = [col.lower().strip() for col in data.colnames]
        
        # Required columns (case insensitive)
        required_cols = ['name', 'ra', 'dec']
        
        # Find the required columns
        col_mapping = {}
        for req_col in required_cols:
            found = False
            
            # Special handling for name column if custom column specified
            if req_col == 'name' and name_column:
                # Look for the user-specified column name
                for i, actual_col in enumerate(data.colnames):
                    if actual_col.lower().strip() == name_column.lower().strip():
                        col_mapping[req_col] = data.colnames[i]
                        found = True
                        print(f"Using '{actual_col}' as name column (user specified)")
                        break
                
                if not found:
                    raise ValueError(f"Specified name column '{name_column}' not found. Available columns: {data.colnames}")
            
            else:
                # Standard column detection
                # First try exact match
                for i, actual_col in enumerate(data_cols_lower):
                    if actual_col == req_col:
                        col_mapping[req_col] = data.colnames[i]
                        found = True
                        break
                
                # If not found, try alternatives
                if not found:
                    alternatives = {
                        'name': ['object', 'source', 'id', 'identifier', 'target', 'obj_name', 'source_name', 'objid', 'object_id', 'source_id'],
                        'ra': ['ra', 'right_ascension', 'alpha', 'r.a.', 'ra_deg', 'ra_j2000', 'raj2000'],
                        'dec': ['dec', 'declination', 'delta', 'de', 'dec_deg', 'dec_j2000', 'dej2000']
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
                if req_col == 'name':
                    raise ValueError(f"Required column '{req_col}' not found. Available columns: {data.colnames}. Use --name-column to specify a different identifier column.")
                else:
                    raise ValueError(f"Required column '{req_col}' not found. Available columns: {data.colnames}")
        
        print(f"Column mapping: {col_mapping}")
        
        # Create a working copy with standardized column names
        # But keep all original columns
        working_data = data.copy()
        
        # Rename the required columns for easier access
        for req_col, actual_col in col_mapping.items():
            if actual_col != req_col:
                # If req_col already exists, use a temp name
                if req_col in working_data.colnames:
                    temp_name = f"_temp_{req_col}"
                    working_data.rename_column(actual_col, temp_name)
                    working_data.rename_column(req_col, f"_orig_{req_col}")
                    working_data.rename_column(temp_name, req_col)
                else:
                    working_data.rename_column(actual_col, req_col)
        
        # Ensure RA and Dec are numeric
        try:
            working_data['ra'] = working_data['ra'].astype(float)
            working_data['dec'] = working_data['dec'].astype(float)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not convert RA/Dec columns to float: {e}")
        
        # Convert name column to strings (handle both text and numeric IDs)
        try:
            # Convert to string, handling various data types
            name_data = working_data['name']
            if hasattr(name_data, 'astype'):
                working_data['name'] = name_data.astype(str)
            else:
                # Handle cases where it's already a list/array
                working_data['name'] = [str(x) for x in name_data]
            
            print(f"Converted name column to string format")
        except Exception as e:
            print(f"Warning: Could not convert name column to string: {e}")
        
        print(f"Successfully processed file with required columns: name, ra, dec")
        return working_data
        
    except Exception as e:
        print(f"Error reading input file: {e}")
        print(f"\nThe file should contain at least these columns (case insensitive):")
        print("- name (or object, source, id, identifier, target) OR use --name-column to specify")
        print("- ra (or right_ascension, alpha, ra_deg)")  
        print("- dec (or declination, delta, dec_deg)")
        print("\nSupported formats:")
        print("- CSV: catalog.csv")
        print("- Text: space-separated, tab-separated files")
        print("- FITS: binary tables in .fits/.fit files")
        print("The file can have additional columns - they will be preserved.")
        print("\nExample usage:")
        print("python check_spherex_coverage.py -i catalog.fits --name-column object_id")
        sys.exit(1)

def query_spherex_count(ra, dec, radius_arcsec=3):
    """
    Query SPHEREx data and return the count of available images
    
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
    count : int
        Number of SPHEREx images found (0 if none)
    """
    try:
        # Create coordinate object
        coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        
        # Query SPHEREx data
        rslt = Irsa.query_sia(pos=(coord.ra, coord.dec, radius_arcsec*u.arcsec), 
                             instrument="SPHEREx")
        
        return len(rslt)
        
    except Exception as e:
        print(f"Error querying SPHEREx data at RA={ra:.6f}, Dec={dec:.6f}: {e}")
        return -1  # Return -1 to indicate query error

def add_spherex_column(data, spherex_counts):
    """
    Add or update the spherex_data column in the table
    """
    if 'spherex_data' in data.colnames:
        # Update existing column
        data['spherex_data'] = spherex_counts
        print("Updated existing 'spherex_data' column")
    else:
        # Add new column
        spherex_col = Column(spherex_counts, name='spherex_data', 
                           description='Number of SPHEREx images available')
        data.add_column(spherex_col)
        print("Added new 'spherex_data' column")
    
    return data

def save_results(data, output_file, input_file):
    """
    Save results to file, preserving original format when possible
    """
    try:
        print(f"Saving results to: {output_file}")
        
        # Determine format based on file extension
        _, out_ext = os.path.splitext(output_file.lower())
        _, in_ext = os.path.splitext(input_file.lower())
        
        if out_ext in ['.fits', '.fit'] or (output_file == input_file and in_ext in ['.fits', '.fit']):
            # FITS format
            try:
                data.write(output_file, format='fits', overwrite=True)
                print(f"Saved as FITS format")
            except Exception as e:
                print(f"Error saving as FITS: {e}")
                # Fallback to CSV
                fallback_name = output_file.replace('.fits', '.csv').replace('.fit', '.csv')
                data.write(fallback_name, format='csv', overwrite=True)
                print(f"Saved as CSV fallback: {fallback_name}")
        
        elif out_ext == '.csv' or ',' in open(input_file, 'r').readline():
            # CSV format
            data.write(output_file, format='csv', overwrite=True)
            print(f"Saved as CSV format")
        else:
            # Space/tab separated format
            ascii.write(data, output_file, format='basic', delimiter=' ', overwrite=True)
            print(f"Saved as space-separated format")
        
        print(f"Successfully saved {len(data)} targets with {len(data.colnames)} columns")
        print(f"Columns: {data.colnames}")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        
        # Fallback: try pandas if available
        try:
            import pandas as pd
            df = data.to_pandas()
            if out_ext == '.csv':
                df.to_csv(output_file, index=False)
            else:
                df.to_csv(output_file, sep=' ', index=False)
            print(f"Saved using pandas fallback")
        except:
            print("Could not save results with any method!")
            # Show the data so user doesn't lose it
            print("First few rows of results:")
            if len(data) > 0:
                print(data[:5])

def print_retry_summary(data, retry_indices, start_time):
    """
    Print summary statistics for retry operations
    """
    spherex_counts = data['spherex_data']
    
    # Check results for retried targets
    retry_results = spherex_counts[retry_indices]
    
    still_errors = np.sum(retry_results == -1)
    now_success = np.sum(retry_results >= 0)
    new_data_found = np.sum(retry_results > 0)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("SPHEREx RETRY SUMMARY")
    print(f"{'='*60}")
    print(f"Error cases retried: {len(retry_indices)}")
    print(f"Now successful: {now_success} ({100*now_success/len(retry_indices):.1f}%)")
    print(f"Still errors: {still_errors} ({100*still_errors/len(retry_indices):.1f}%)")
    print(f"Found new data: {new_data_found}")
    
    if new_data_found > 0:
        new_images = np.sum(retry_results[retry_results > 0])
        print(f"Total new images found: {new_images}")
    
    print(f"Retry processing time: {elapsed_time:.1f} seconds")
    print(f"Average time per retry: {elapsed_time/len(retry_indices):.2f} seconds")
    
    # Overall file statistics
    all_valid = spherex_counts[spherex_counts >= 0]
    all_errors = np.sum(spherex_counts == -1)
    
    print(f"\nOVERALL FILE STATUS:")
    print(f"Total targets: {len(data)}")
    print(f"Successful queries: {len(all_valid)} ({100*len(all_valid)/len(data):.1f}%)")
    print(f"Remaining errors: {all_errors} ({100*all_errors/len(data):.1f}%)")

def print_summary(data, start_time):
    """
    Print summary statistics
    """
    if 'spherex_data' not in data.colnames:
        return
    
    spherex_counts = data['spherex_data']
    
    # Filter out error cases (-1)
    valid_counts = spherex_counts[spherex_counts >= 0]
    error_count = np.sum(spherex_counts == -1)
    
    with_data = np.sum(valid_counts > 0)
    without_data = np.sum(valid_counts == 0)
    total_images = np.sum(valid_counts)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("SPHEREx COVERAGE SUMMARY")
    print(f"{'='*60}")
    print(f"Total targets processed: {len(data)}")
    print(f"Targets with SPHEREx data: {with_data} ({100*with_data/len(valid_counts):.1f}%)")
    print(f"Targets without SPHEREx data: {without_data} ({100*without_data/len(valid_counts):.1f}%)")
    print(f"Total SPHEREx images found: {total_images}")
    
    if with_data > 0:
        avg_images = total_images / with_data
        max_images = np.max(valid_counts)
        print(f"Average images per target (with data): {avg_images:.1f}")
        print(f"Maximum images at single position: {max_images}")
    
    if error_count > 0:
        print(f"Query errors: {error_count}")
    
    print(f"Processing time: {elapsed_time:.1f} seconds")
    print(f"Average time per target: {elapsed_time/len(data):.2f} seconds")
    
    # Show some examples
    if with_data > 0:
        print(f"\nTargets with most SPHEREx coverage:")
        sorted_indices = np.argsort(valid_counts)[::-1]
        for i in range(min(3, with_data)):
            idx = sorted_indices[i]
            if valid_counts[idx] > 0:
                print(f"  {data['name'][idx]}: {valid_counts[idx]} images")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='''
        Check SPHEREx data availability for a list of coordinates.
        
        This script queries the SPHEREx archive to determine how many images
        are available at each target position. Results are saved with a new
        'spherex_data' column containing the image count.
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES)

    parser.add_argument('-i', '--input', required=True, type=str,
                       help='Input file containing name, ra, dec columns')

    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output file (default: update input file)')

    parser.add_argument('--radius', type=float, default=3.0,
                       help='Search radius in arcseconds (default: 3.0)')

    parser.add_argument('--name-column', type=str, default=None,
                       help='Column name to use as object identifier (e.g., "object_id", "source_id")')

    parser.add_argument('--delimiter', type=str, default=None,
                       help='Column delimiter in input file (default: auto-detect)')

    parser.add_argument('--progress', type=int, default=50,
                       help='Show progress every N targets (default: 50)')

    parser.add_argument('--update', action='store_true',
                       help='Update input file in place (same as -o input_file)')

    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually querying')

    parser.add_argument('--retry-errors', action='store_true',
                       help='Only retry sources with spherex_data = -1 (query errors)')

    parser.add_argument('--start-from', type=int, default=0,
                       help='Start processing from target N (for resuming)')

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        sys.exit(1)
    
    # Determine output file
    if args.retry_errors:
        output_file = args.input  # Always update original file in retry mode
        print(f"Retry mode: will update {args.input} in place")
    elif args.update:
        output_file = args.input
    elif args.output:
        output_file = args.output
    else:
        # Default: add suffix to input filename
        base, ext = os.path.splitext(args.input)
        output_file = f"{base}_spherex{ext}"
    
    print(f"SPHEREx Coverage Checker")
    print(f"Input file: {args.input}")
    if not args.retry_errors:
        print(f"Output file: {output_file}")
    print(f"Search radius: {args.radius}\"")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Read input file
    data = read_input_file(args.input, delimiter=args.delimiter, name_column=args.name_column)
    
    # Handle retry-errors mode
    if args.retry_errors:
        if 'spherex_data' not in data.colnames:
            print("Error: --retry-errors specified but no 'spherex_data' column found")
            print("Run without --retry-errors first to create the column")
            sys.exit(1)
        
        error_indices = np.where(data['spherex_data'] == -1)[0]
        if len(error_indices) == 0:
            print("No error cases (spherex_data = -1) found. Nothing to retry.")
            return
        
        print(f"Found {len(error_indices)} error cases to retry")
        target_indices = error_indices
        output_file = args.input  # Always update original file in retry mode
    else:
        target_indices = np.arange(len(data))
    
    if args.dry_run:
        if args.retry_errors:
            print(f"\nDRY RUN: Would retry {len(target_indices)} error cases")
        else:
            print(f"\nDRY RUN: Would check {len(data)} targets for SPHEREx coverage")
        print(f"Would save results to: {output_file}")
        return
    
    # Initialize results array
    if args.retry_errors:
        # Keep existing data, only update error cases
        spherex_counts = np.array(data['spherex_data'])
        start_idx = 0  # We'll process specific indices, not sequential
        total_to_process = len(target_indices)
    else:
        spherex_counts = np.full(len(data), -1, dtype=int)
        
        # Check if we already have spherex_data column (for resuming)
        start_idx = args.start_from
        if 'spherex_data' in data.colnames and start_idx == 0:
            # Look for first unprocessed target (spherex_data == -1 or missing)
            existing_data = data['spherex_data']
            unprocessed = np.where(existing_data < 0)[0]
            if len(unprocessed) > 0:
                start_idx = unprocessed[0]
                spherex_counts = np.array(existing_data)
                print(f"Found existing spherex_data column, resuming from target {start_idx}")
        
        target_indices = target_indices[start_idx:]
        total_to_process = len(target_indices)
    
    # Process each target
    start_time = time.time()
    
    if args.retry_errors:
        print(f"Retrying {total_to_process} error cases...")
        retry_success_count = 0
        retry_fail_count = 0
        
        for idx, i in enumerate(target_indices):
            row = data[i]
            name = str(row['name']).strip()
            ra = float(row['ra'])
            dec = float(row['dec'])
            
            # Query SPHEREx data
            count = query_spherex_count(ra, dec, args.radius)
            spherex_counts[i] = count
            
            # Track retry results
            if count >= 0:
                retry_success_count += 1
            else:
                retry_fail_count += 1
            
            # Show progress
            if (idx + 1) % args.progress == 0 or idx == total_to_process - 1:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (total_to_process - idx - 1) / rate if rate > 0 else 0
                success_rate = 100 * retry_success_count / (idx + 1) if (idx + 1) > 0 else 0
                print(f"Retry progress: {idx+1}/{total_to_process} ({100*(idx+1)/total_to_process:.1f}%) - "
                      f"Rate: {rate:.1f}/s - ETA: {eta:.0f}s - "
                      f"Fixed: {retry_success_count} - Still failing: {retry_fail_count} ({success_rate:.1f}% success) - "
                      f"Current: {name}")
            
            # Save intermediate results every 50 retries (more frequent for problematic cases)
            if (idx + 1) % 50 == 0:
                temp_data = data.copy()
                temp_data = add_spherex_column(temp_data, spherex_counts)
                save_results(temp_data, output_file, args.input)
                print(f"Saved intermediate retry results at {idx+1}/{total_to_process} retries")
            
            # Brief pause to be nice to the server
            time.sleep(0.1)
    
    else:
        for idx, i in enumerate(target_indices):
            row = data[i]
            name = str(row['name']).strip()
            ra = float(row['ra'])
            dec = float(row['dec'])
            
            # Show progress
            if (idx + 1) % args.progress == 0 or idx == total_to_process - 1:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (total_to_process - idx - 1) / rate if rate > 0 else 0
                actual_target_num = i + 1 if not args.retry_errors else i + 1
                print(f"Progress: {actual_target_num}/{len(data)} ({100*actual_target_num/len(data):.1f}%) - "
                      f"Rate: {rate:.1f} targets/s - ETA: {eta:.0f}s - Current: {name}")
            
            # Query SPHEREx data
            count = query_spherex_count(ra, dec, args.radius)
            spherex_counts[i] = count
            
            # Brief pause to be nice to the server
            time.sleep(0.1)
            
            # Save intermediate results every 100 targets
            if (idx + 1) % 100 == 0:
                temp_data = data.copy()
                temp_data = add_spherex_column(temp_data, spherex_counts)
                save_results(temp_data, output_file, args.input)
                print(f"Saved intermediate results at target {idx+1}")
    
    # Add/update the spherex_data column
    data = add_spherex_column(data, spherex_counts)
    
    # Save final results
    save_results(data, output_file, args.input)
    
    # Print summary
    if args.retry_errors:
        print_retry_summary(data, target_indices, start_time)
    else:
        print_summary(data, start_time)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()