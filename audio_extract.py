import subprocess
import tempfile
from pathlib import Path

def extract_audio(video_path):
    video_path = str(video_path)

    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",              # 不要视频
        "-ac", "1",         # 单声道
        "-ar", "16000",     # 采样率
        "-loglevel", "error",
        audio_path
    ]

    subprocess.run(cmd)

    return audio_path