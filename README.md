# 🔧 Tools - Useful Audio & Video Utilities

This repository contains handy scripts for audio and video processing using `ffmpeg`. These tools can help automate repetitive tasks like format conversion and video splitting.

---

## 📁 Contents

- [`mp3_to_m4a.py`](#-mp3_to_m4apy) — Convert MP3 files to M4A format using AAC codec with metadata preservation.
- [`spilt_video.py`](#-spilt_videopy) — Split video files into equal-length segments without re-encoding the video, and re-encode audio at 320kbps.

---

## 🎵 `mp3_to_m4a.py`
## 🎵 `m4a_to_mp3.py`
## 🎵 `Spilt_video.py`

### 🔄 Description:
This script converts all `.mp3` files in a given directory (including subfolders) into `.m4a` format using the AAC codec. It also preserves ID3 metadata.

### ⚙️ Features:
- Preserves metadata (`-map_metadata 0`)
- Converts in parallel using multiple threads
- Customizable audio bitrate (default `320k`)
- Supports overwrite mode




#### only readme made my ChatGpt

