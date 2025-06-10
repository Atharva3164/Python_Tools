import os
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

input_dir = Path("")     # ← Set your input M4A folder path here
output_dir = Path("")    # ← Set your desired output MP3 folder path here

def convert_file(args):
    m4a_path, bitrate, overwrite = args
    relative_path = m4a_path.relative_to(input_dir)
    mp3_path = output_dir / relative_path.with_suffix('.mp3')

    if mp3_path.exists() and not overwrite:
        return {'status': 'skipped', 'file': m4a_path.name}

    try:
        mp3_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            'ffmpeg',
            '-i', str(m4a_path),
            '-map_metadata', '0',
            '-id3v2_version', '3',
            '-c:a', 'libmp3lame',
            '-b:a', bitrate,
            '-loglevel', 'error',
            '-y' if overwrite else '-n',
            str(mp3_path)
        ]
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        return {'status': 'success', 'file': m4a_path.name}
    except subprocess.CalledProcessError as e:
        return {'status': 'error', 'file': m4a_path.name, 'message': e.stderr.decode()}
    except Exception as e:
        return {'status': 'error', 'file': m4a_path.name, 'message': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Convert M4A to MP3 with metadata preservation')
    parser.add_argument('-b', '--bitrate', type=str, default='320k',
                        help='MP3 bitrate (e.g., 192k, 256k, 320k)')
    parser.add_argument('-t', '--threads', type=int, default=os.cpu_count(),
                        help='Number of parallel threads')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Overwrite existing files')
    args = parser.parse_args()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} not found")

    output_dir.mkdir(exist_ok=True)

    m4a_files = list(input_dir.rglob('*.m4a'))
    if not m4a_files:
        print("No M4A files found in input directory")
        return

    print(f"Found {len(m4a_files)} M4A files")
    print(f"Using {args.threads} parallel threads")
    print(f"Output format: MP3 @ {args.bitrate}")
    print(f"Output folder: {output_dir}")

    tasks = [(file, args.bitrate, args.overwrite) for file in m4a_files]

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        results = list(tqdm(executor.map(convert_file, tasks),
                            total=len(m4a_files),
                            desc="Converting files"))

    success = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = [r for r in results if r['status'] == 'error']

    print("\nConversion complete:")
    print(f" - Success: {success}")
    print(f" - Skipped: {skipped}")
    print(f" - Errors: {len(errors)}")

    if errors:
        print("\nError details:")
        for error in errors[:3]:
            print(f"File: {error['file']}")
            print(f"Error: {error['message']}\n")

if __name__ == '__main__':
    main()

print("Done.................................")
