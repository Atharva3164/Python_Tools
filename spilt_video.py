# pip install ffmpeg
# pip install libavcodec-extra

from pathlib import Path 
import subprocess          

# ─────────────────────────────────────────────────
folder_path    = r"path"  # paste your folder path here (must contain your video in the path folder) not outside the ffolder
segment_length = 40       # segment length in seconds 
# ───────────────────────────────────────────────────────────────────────────────

def get_duration(path: Path) -> float:

    cmd = [
        "ffprobe", "-i", str(path),
        "-show_entries", "format=duration", "-v", "quiet",
        "-of", "csv=p=0"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)  
    return float(result.stdout.strip())

def split_video(input_path: Path, output_dir: Path, seg_len: int):
    """
    Split one video into seg_len‑second clips, re-encoding audio at 320kbps.
    """
    duration = get_duration(input_path)  # total seconds
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = output_dir / (input_path.stem + "_%03d" + input_path.suffix)

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-c:v", "copy",            
        "-c:a", "aac",              
        "-b:a", "320k",          
        "-map", "0",
        "-f", "segment",
        "-segment_time", str(seg_len),
        "-reset_timestamps", "1",
        str(out_pattern)
    ]
    subprocess.run(cmd, check=True)


def main():
    folder = Path(folder_path)   
    exts = [".mp4", ".mov", ".avi", ".mkv"]
    for vid in folder.iterdir():
        if vid.suffix.lower() in exts:
            outdir = folder / f"{vid.stem}_segments"
            print(f"Splitting {vid.name} into {segment_length}s parts…")
            split_video(vid, outdir, segment_length)
            print(f"→ Done: {outdir}\n")

if __name__ == "__main__":
    main()
