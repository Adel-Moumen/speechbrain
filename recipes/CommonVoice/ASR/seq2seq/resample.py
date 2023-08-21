"""
This script is designed to resample audio files in a folder recursively, providing the option to choose between two audio processing backends: FFmpeg or SoX.
It also preserves the source folder structure during the resampling process and automatically converts audio files to the WAV format and PCM mono channel during resampling.
To run this script, do the following:
 python resample.py --input_folder /path/to/input_folder --output_folder /path/to/output_folder
Authors
 * Jarod Duret 2023
"""

import argparse
import pathlib as pl
import subprocess as sp
from speechbrain.utils.parallel import parallel_map
from tqdm import tqdm
import math
import functools

def resample(filename, output_sr, input_folder, output_folder, audio_backend):
    """
    Resample a single audio file using the specified backend.
    Arguments
    ---------
    func_args (tuple)
        A tuple containing the following elements:
            - filename (Path)
                Path to the input audio file.
            - output_sr (int)
                Desired sample rate for resampling.
            - input_folder (Path)
                Path to the input directory.
            - output_folder (Path)
                Path to the output directory.
            - audio_backend (str)
                Chosen audio processing backend ("ffmpeg" or "sox").
    """
    
    # filename, output_sr, input_folder, output_folder, audio_backend = func_args
    filename = pl.Path(filename)
    relative_path = filename.relative_to(input_folder)
    out_file = output_folder / relative_path.parent / f"{filename.stem}.wav"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if audio_backend == "ffmpeg":
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            filename.as_posix(),
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(output_sr),
            "-ac",
            "1",
            out_file,
        ]
    elif audio_backend == "sox":
        cmd = [
            "sox",
            filename.as_posix(),
            "-r",
            str(output_sr),
            "-b",
            "16",
            "-c",
            "1",
            out_file,
        ]
    else:
        raise ValueError("Invalid audio backend specified")
    sp.run(cmd, stdout=sp.DEVNULL, stderr=sp.STDOUT, text=False)

def resample_folder(
    input_folder,
    output_sr,
    output_folder=None,
    file_ext="wav",
    n_jobs=10,
    audio_backend="ffmpeg",
):
    """
    Resample audio files within a directory and its subdirectories.
    Arguments
    ---------
    input_folder (str)
        Path to the folder containing the audio files.
    output_sr (int)
        Desired sample rate for resampling.
    output_folder (str)
        Path to the output directory. If not defined, operation is done in place.
    file_ext (str)
        Extension of the audio files to be resampled. Default is "wav".
    n_jobs (int)
        Number of threads to use. Default is 10.
    audio_backend (str)
        Audio processing backend to use ("ffmpeg" or "sox"). Default is "ffmpeg".
    """
    input_folder = pl.Path(input_folder)
    output_folder = pl.Path(output_folder) if output_folder else input_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    # I think we can remove this part. See ligne 39-43
    # print("Copying directory structure...")
    # for sub_dir in input_folder.glob("**/*"):
    #     if sub_dir.is_dir():
    #         relative_sub_dir = sub_dir.relative_to(input_folder)
    #         target_sub_dir = output_folder / relative_sub_dir
    #         target_sub_dir.mkdir(parents=True, exist_ok=True)
    print("Resampling the audio files...")
    audio_files = list(input_folder.glob(f"**/*.{file_ext}"))
    resample_processor = functools.partial(
        resample,
        output_sr=output_sr,
        input_folder=input_folder,
        output_folder=output_folder,
        audio_backend=audio_backend,
    )
    for row in parallel_map(resample_processor, audio_files, n_jobs):
        if row is None:
            continue
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Resample audio files in a folder recursively using FFmpeg or SoX.\n\n
                       Example run
                            python resample.py
                                --input_folder /root/LJSpeech/
                                --output_sr 16000
                                --output_folder /root/LJSpeech_16k/
                                --file_ext wav
                                --n_jobs 24
                                --audio_backend ffmpeg
                    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path of the folder containing the audio files to resample",
    )
    parser.add_argument(
        "--output_sr",
        type=int,
        default=22050,
        required=False,
        help="Sample rate to which the audio files should be resampled",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=False,
        help="Path of the destination folder. If not defined, the operation is done in place",
    )
    parser.add_argument(
        "--file_ext",
        type=str,
        default="wav",
        required=False,
        help="Extension of the audio files to resample",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Number of threads to use, by default it uses all available cores",
    )
    parser.add_argument(
        "--audio_backend",
        type=str,
        default="ffmpeg",
        choices=["ffmpeg", "sox"],
        help="Audio processing backend: ‘ffmpeg’ or ‘sox’",
    )
    args = parser.parse_args()
    resample_folder(
        args.input_folder,
        args.output_sr,
        args.output_folder,
        args.file_ext,
        args.n_jobs,
        args.audio_backend,
    )