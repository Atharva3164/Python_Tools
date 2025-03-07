import os
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# User provides input directory path directly
input_dir = Path("C:/ATHARVA/Atharva/Music")  # Change this to your desired folder
output_dir = Path("C:/ATHARVA/Atharva/Music/M4A_converted")  # Output folder

def convert_file(args):
    mp3_path, bitrate, overwrite = args
    relative_path = mp3_path.relative_to(input_dir)
    m4a_path = output_dir / relative_path.with_suffix('.m4a')
    
    if m4a_path.exists() and not overwrite:
        return {'status': 'skipped', 'file': mp3_path.name}
    
    try:
        m4a_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            'ffmpeg',
            '-i', str(mp3_path),
            '-map_metadata', '0',
            '-id3v2_version', '3',
            '-c:a', 'aac',
            '-b:a', bitrate,
            '-c:v', 'copy',
            '-loglevel', 'error',
            '-y' if overwrite else '-n',
            str(m4a_path)
        ]
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        return {'status': 'success', 'file': mp3_path.name}
    except subprocess.CalledProcessError as e:
        return {'status': 'error', 'file': mp3_path.name, 'message': e.stderr.decode()}
    except Exception as e:
        return {'status': 'error', 'file': mp3_path.name, 'message': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Convert MP3 to M4A with metadata preservation')
    parser.add_argument('-b', '--bitrate', type=str, default='320k', 
                       help='AAC bitrate (e.g., 192k, 256k, 320k)')
    parser.add_argument('-t', '--threads', type=int, default=os.cpu_count(),
                       help='Number of parallel threads')
    parser.add_argument('-o', '--overwrite', action='store_true',
                       help='Overwrite existing files')
    args = parser.parse_args()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} not found")

    output_dir.mkdir(exist_ok=True)  # Ensure the output directory exists

    mp3_files = list(input_dir.rglob('*.mp3'))
    if not mp3_files:
        print("No MP3 files found in input directory")
        return

    print(f"Found {len(mp3_files)} MP3 files")
    print(f"Using {args.threads} parallel threads")
    print(f"Output format: M4A @ {args.bitrate}")
    print(f"Output folder: {output_dir}")

    tasks = [(file, args.bitrate, args.overwrite) for file in mp3_files]

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        results = list(tqdm(executor.map(convert_file, tasks), 
                      total=len(mp3_files), 
                      desc="Converting files"))

    # Process results
    success = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = [r for r in results if r['status'] == 'error']

    print("\nConversion complete:")
    print(f" - Success: {success}")
    print(f" - Skipped: {skipped}")
    print(f" - Errors: {len(errors)}")

    if errors:
        print("\nError details:")
        for error in errors[:3]:  # Show first 3 errors
            print(f"File: {error['file']}")
            print(f"Error: {error['message']}\n")

if __name__ == '__main__':
    main()

print("Done..")