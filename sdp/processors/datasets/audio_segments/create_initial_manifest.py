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
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm

from tqdm.contrib.concurrent import process_map

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import download_file, extract_archive

from sdp.processors.datasets.audio_segments.audio_modification import construct_audio

class CreateInitialManifestAudioSegments(BaseParallelProcessor):

    def __init__(
        self,
        audio_dir: str,
        combined_audio_dir: str,
        max_duration: int = 20, 
        min_duration: int = 5, 
        start_idx: int = 0,
        end_idx: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.audio_dir = audio_dir
        self.combined_audio_dir = combined_audio_dir

        self.max_duration = max_duration
        self.min_duration = min_duration

        self.start_idx = start_idx
        self.end_idx = end_idx

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

    def process_dataset_entry(self, data_entry: Dict):

        last_first_entries = []
        parent_id, segments = list(data_entry.items())[0]

        for i in range(len(segments) - 1):
            if segments[i]['durations'][-1] > self.min_duration or len(segments[i]['combined_ids']) == 1:

                de_last = {'audio_filepath': Path(self.audio_dir, parent_id, segments[i]['combined_ids'][-1]).with_suffix('.opus').as_posix(),
                            'duration': segments[i]['durations'][-1],
                            'label': 'id',
                            'parent_id': parent_id}

            else:
                combined_audio_path, combined_dur = construct_audio(filenames=segments[i]['combined_ids'],
                                                                    durations=segments[i]['durations'],
                                                                    parent_id=parent_id,
                                                                    min_duration=self.min_duration,
                                                                    audio_dir=self.audio_dir, 
                                                                    combined_audio_dir=self.combined_audio_dir,
                                                                    end=True)

                de_last = {'audio_filepath': combined_audio_path,
                            'duration': combined_dur,
                            'label': 'id',
                            'parent_id': parent_id}
            
            last_first_entries.append(DataEntry(de_last))

            if len(segments[i+1]['combined_ids']) > 1 or i == len(segments) - 2:
                if segments[i+1]['durations'][0] > self.min_duration or i == len(segments) - 2:
                    de_first = {'audio_filepath': Path(self.audio_dir, parent_id, segments[i+1]['combined_ids'][0]).with_suffix('.opus').as_posix(),
                                'duration': segments[i+1]['durations'][0],
                                'label': 'id',
                                'parent_id': parent_id}

                else:
                    combined_audio_path, combined_dur = construct_audio(filenames=segments[i+1]['combined_ids'],
                                                                        durations=segments[i+1]['durations'],
                                                                        parent_id=parent_id,
                                                                        min_duration=self.min_duration,
                                                                        audio_dir=self.audio_dir,
                                                                        combined_audio_dir=self.combined_audio_dir)


                    de_first = {'audio_filepath': combined_audio_path,
                                'duration': combined_dur,
                                'label': 'id',
                                'parent_id': parent_id}



                last_first_entries.append(DataEntry(de_first))

        return last_first_entries


    

