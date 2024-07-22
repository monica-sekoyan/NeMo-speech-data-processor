# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
from pathlib import Path

import soundfile as sf
import sox
from pydub import AudioSegment
from sox import Transformer

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class x(BaseParallelProcessor):
    """Processor to create initial manifest for the VoxPopuli dataset.

    Dataset link: https://github.com/facebookresearch/voxpopuli/

    Downloads and unzips raw VoxPopuli data for the specified language,
    and creates an initial manifest using the transcripts provided in the
    raw data.

    .. note::
        This processor will install a couple of Python packages, including
        PyTorch, so it might be a good idea to run it in an isolated Python
        environment.

    Args:
        raw_data_dir (str): the directory where the downloaded data will be/is saved.
        language_id (str): the language of the data you wish to be downloaded.
            E.g., "en", "es", "it", etc.
        data_split (str): "train", "dev" or "test".
        resampled_audio_dir (str): the directory where the resampled wav
            files will be stored.
        target_samplerate (int): sample rate (Hz) to use for resampling.
            Defaults to 16000.
        target_nchannels (int): number of channels to create during resampling process.
            Defaults to 1.

    Returns:
        This processor generates an initial manifest file with the following fields::

            {
                "audio_filepath": <path to the audio file>,
                "duration": <duration of the audio in seconds>,
                "text": <transcription (with provided normalization)>,
                "raw_text": <original transcription (without normalization)>,
                "speaker_id": <speaker id>,
                "gender": <speaker gender>,
                "age": <speaker age>,
                "is_gold_transcript": <whether the transcript has been verified>,
                "accent": <speaker accent, if known>,
            }
    """

    def __init__(
        self,
        raw_data_dir: str,
        data_type: str,
        resampled_audio_dir: str,
        data_split: str = 'training',
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.data_type = data_type
        self.resampled_audio_dir = resampled_audio_dir
        self.data_split = data_split
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels

        self.audios_dir = Path(self.raw_data_dir, self.data_type, self.data_split, 'audio')
        self.transcriptions_dir = Path(self.raw_data_dir, self.data_type, self.data_split, 'transcription')
        self.demographics_file = Path(self.raw_data_dir, self.data_type, 'reference_matetials', 'demographics.tsv')

        if not os.path.exists(self.resampled_audio_dir):
            os.makedirs(self.resampled_audio_dir, exist_ok=True)

    def read_manifest(self):
        """Reading the input manifest file.

        .. note::
            This function should be overridden in the "initial" class creating
            manifest to read from the original source of data.
        """

        with open(self.demographics_file, "rt", encoding="utf8") as fin:
            header = fin.readline()
            titles = [t.strip() for t in header.split('\t')]
            for line in fin:
                data_entry = dict(zip(titles, line.split('\t')))
                yield data_entry

    def process_dataset_entry(self, data_entry: dict):
        audio_filepath = Path(self.audios_dir, data_entry[self.outputFn])
        transcription_path = Path(self.transcriptions_dir, data_entry[self.outputFn].stem).with_suffix('.txt')

        tgt_audio_filepath = (
            Path(self.resampled_audio_dir, data_entry[self.outputFn].stem).with_suffix('.flac').as_posix()
        )

        try:
            audio = AudioSegment.from_file(audio_filepath)

            if audio.frame_rate != self.target_samplerate:
                audio = audio.set_frame_rate(self.target_samplerate)

            if audio.channels != self.target_nchannels:
                return NotImplementedError

            with open(transcription_path, 'rt') as f:
                timestamps = f[::2]
                texts = f[1::2]

            data_entries = []

            for idx in range(len(timestamps) - 1):
                new_audio_filepath = tgt_audio_filepath.replace('.flac', f'_{idx}.flac')

                start = float(timestamps[idx])
                end = float(timestamps[idx + 1])

                audio_segment = audio.get_array_of_samples()[
                    start * self.target_samplerate : end * self.target_samplerate
                ]

                if not os.path.exists(new_audio_filepath):
                    audio_segment.export(tgt_audio_filepath, format="flac")

                modified_entry = data_entry.copy()
                modified_entry['audio_filepath'] = new_audio_filepath
                modified_entry['text'] = texts[idx]
                modified_entry['duration'] = end - start

                data_entries.append(DataEntry(data=modified_entry))

        except Exception as e:
            print(data_entry)
            raise e

        return data_entries
