# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# To convert mp3 files to wav using sox, you must have installed sox with mp3 support
# For example sudo apt-get install libsox-fmt-mp3
import csv
import glob
import os
import json
import itertools
import librosa
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from pydub import AudioSegment
import soundfile as sf

from tqdm.contrib.concurrent import process_map

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import download_file, extract_archive

from sdp.processors.datasets.audio_segments.audio_modification import construct_audio

class ConvertAudiosToFlac(BaseParallelProcessor):

    def __init__(
        self,
        audio_dir: str,
        new_audio_dir: str,
        start_idx: int = 0,
        end_idx: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.audio_dir = audio_dir
        self.new_audio_dir = new_audio_dir

        self.start_idx = start_idx
        self.end_idx = end_idx

        self.corrupted_audios = []

    def read_manifest(self):
        if self.input_manifest_file is None:
            raise NotImplementedError("Override this method if the processor creates initial manifest")

        with open(self.input_manifest_file, "rt", encoding="utf8") as fin:
            for idx, line in enumerate(fin):
                if idx < self.start_idx:
                    continue 
                if self.end_idx and idx == self.end_idx:
                    break
                yield json.loads(line)

    def process_dataset_entry(self, data_entry):

        audio_filepath = Path(self.audio_dir, data_entry['audio_filepath'])

        new_audio_dir = Path(self.new_audio_dir, data_entry['parent_id'])
        new_audio_dir.mkdir(exist_ok=True, parents=True)
        new_audio_filepath = Path(new_audio_dir, data_entry['id']).with_suffix('.flac')

        # if new_audio_filepath.exists():
        #     data_entry['audio_filepath'] = new_audio_filepath.as_posix()
        #     return [DataEntry(data_entry)]

        try:

            audio, sr = librosa.load(audio_filepath)
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            # sf.write(new_audio_filepath, audio, 16000)

            data_entry['audio_filepath'] = new_audio_filepath.as_posix()
            data_entry['duration'] = librosa.get_duration(y=audio, sr=16000)

            return [DataEntry(data_entry)]

        except Exception as e:
            self.corrupted_audios.append(audio_filepath)
            return [DataEntry(None)]


    

