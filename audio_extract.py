import shutil
import subprocess
import tempfile
from pathlib import Path


def _resolve_ffmpeg():
    """找 ffmpeg 可执行文件：优先系统 PATH，找不到就回退 pip 装的 imageio-ffmpeg
    自带的静态二进制（无需管理员权限 / 不用改系统 PATH）。"""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "找不到 ffmpeg：系统 PATH 里没有，且 imageio-ffmpeg 也不可用。"
            "请 `pip install imageio-ffmpeg` 或自行安装 ffmpeg。"
        ) from e


def extract_audio(video_path):
    """从视频抽单声道 16k wav（emotion2vec / wav2vec2 都要 16k）。

    RAVDESS 是 .mp4，音频藏在容器里，soundfile/torchaudio 读不了，必须 ffmpeg 解码。
    """
    video_path = str(video_path)

    ffmpeg = _resolve_ffmpeg()
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    cmd = [
        ffmpeg,
        "-y",
        "-i", video_path,
        "-vn",              # 不要视频
        "-ac", "1",         # 单声道
        "-ar", "16000",     # 采样率
        "-loglevel", "error",
        audio_path,
    ]

    # capture_output 便于失败时把 ffmpeg 的 stderr 带出来；check=False 手动判定
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not Path(audio_path).exists() or Path(audio_path).stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg 抽音频失败 (exit={proc.returncode}) for {video_path}: "
            f"{proc.stderr.strip() or 'no stderr'}"
        )

    return audio_path
