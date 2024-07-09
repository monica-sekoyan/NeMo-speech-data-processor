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
import itertools
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from sdp.logging import logger
from sdp.processors.base_processor import (
    BaseParallelProcessor,
    BaseProcessor,
    DataEntry,
)

# from sdp.processors.datasets.audio_segments.audio_modification import construct_audio
from sdp.utils.common import download_file, extract_archive

# class CreateInitialManifestAudioSegments(BaseParallelProcessor):
#     def __init__(
#         self,
#         audio_dir: str,
#         combined_audio_dir: str,
#         max_duration: int = 20,
#         min_duration: int = 5,
#         start_idx: int = 0,
#         end_idx: int = None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)

#         self.audio_dir = audio_dir
#         self.combined_audio_dir = combined_audio_dir

#         self.max_duration = max_duration
#         self.min_duration = min_duration

#         self.start_idx = start_idx
#         self.end_idx = end_idx

#     def read_manifest(self):
#         if self.input_manifest_file is None:
#             raise NotImplementedError("Override this method if the processor creates initial manifest")

#         with open(self.input_manifest_file, "rt", encoding="utf8") as fin:
#             for idx, line in enumerate(fin):
#                 if idx < self.start_idx:
#                     continue
#                 if self.end_idx and idx == self.end_idx:
#                     break
#                 yield json.loads(line)

#     def process_dataset_entry(self, data_entry: Dict):
#         last_first_entries = []
#         parent_id, segments = list(data_entry.items())[0]

#         for i in range(len(segments) - 1):
#             if segments[i]['durations'][-1] > self.min_duration or len(segments[i]['combined_ids']) == 1:
#                 de_last = {
#                     'audio_filepath': Path(self.audio_dir, parent_id, segments[i]['combined_ids'][-1])
#                     .with_suffix('.opus')
#                     .as_posix(),
#                     'duration': segments[i]['durations'][-1],
#                     'label': 'id',
#                     'parent_id': parent_id,
#                 }

#             else:
#                 combined_audio_path, combined_dur = construct_audio(
#                     filenames=segments[i]['combined_ids'],
#                     durations=segments[i]['durations'],
#                     parent_id=parent_id,
#                     min_duration=self.min_duration,
#                     audio_dir=self.audio_dir,
#                     combined_audio_dir=self.combined_audio_dir,
#                     end=True,
#                 )

#                 de_last = {
#                     'audio_filepath': combined_audio_path,
#                     'duration': combined_dur,
#                     'label': 'id',
#                     'parent_id': parent_id,
#                 }

#             last_first_entries.append(DataEntry(de_last))

#             if len(segments[i + 1]['combined_ids']) > 1 or i == len(segments) - 2:
#                 if segments[i + 1]['durations'][0] > self.min_duration or i == len(segments) - 2:
#                     de_first = {
#                         'audio_filepath': Path(self.audio_dir, parent_id, segments[i + 1]['combined_ids'][0])
#                         .with_suffix('.opus')
#                         .as_posix(),
#                         'duration': segments[i + 1]['durations'][0],
#                         'label': 'id',
#                         'parent_id': parent_id,
#                     }

#                 else:
#                     combined_audio_path, combined_dur = construct_audio(
#                         filenames=segments[i + 1]['combined_ids'],
#                         durations=segments[i + 1]['durations'],
#                         parent_id=parent_id,
#                         min_duration=self.min_duration,
#                         audio_dir=self.audio_dir,
#                         combined_audio_dir=self.combined_audio_dir,
#                     )

#                     de_first = {
#                         'audio_filepath': combined_audio_path,
#                         'duration': combined_dur,
#                         'label': 'id',
#                         'parent_id': parent_id,
#                     }

#                 last_first_entries.append(DataEntry(de_first))

#         return last_first_entries


class CustomDataSplitSUNO(BaseParallelProcessor):
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
        data_split: str,
        resampled_audio_dir: str,
        dev_ratio: float = 0.02,
        test_ratio: float = 0.02,
        shuffle_seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_split = data_split
        self.resampled_audio_dir = resampled_audio_dir
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
        self.shuffle_seed = shuffle_seed

    def prepare(self):
        self.split_dir = f'{self.resampled_audio_dir}/{self.data_split}'
        os.makedirs(self.split_dir, exist_ok=True)

    def read_manifest(self):
        random.seed(self.shuffle_seed)

        with open(self.input_manifest_file, 'r') as fin:
            dataset_entries = [json.loads(line) for line in fin.readlines()]

        random.shuffle(dataset_entries)

        if self.data_split == 'test':
            dataset_entries = dataset_entries[-int(self.test_ratio * len(dataset_entries)) :]

        elif self.data_split == 'dev':
            dataset_entries = dataset_entries[
                -(int(self.test_ratio * len(dataset_entries)) + int(self.dev_ratio * len(dataset_entries))) : -int(
                    self.test_ratio * len(dataset_entries)
                )
            ]
        elif self.data_split == 'train':
            dataset_entries = dataset_entries[
                : -(int(self.test_ratio * len(dataset_entries)) + int(self.dev_ratio * len(dataset_entries)))
            ]

        # print('TEST: ', "[", -int(self.test_ratio * len(dataset_entries)), " ,:)")
        # print('DEV: ', "[", -(int(self.test_ratio * len(dataset_entries)) + int(self.dev_ratio * len(dataset_entries))),
        #       " ,",
        #       -int(
        #             self.test_ratio * len(dataset_entries)),
        #       ")")

        # print('TRAIN: ', "(:, ",  -(int(self.test_ratio * len(dataset_entries)) + int(self.dev_ratio * len(dataset_entries))), ")")

        return dataset_entries

    def process_dataset_entry(self, data_entry: str):
        shutil.copy(data_entry['audio_filepath'], self.split_dir)

        data_entry['audio_filepath'] = Path(self.split_dir, Path(data_entry['audio_filepath']).name).as_posix()
        return [DataEntry(data=data_entry)]
