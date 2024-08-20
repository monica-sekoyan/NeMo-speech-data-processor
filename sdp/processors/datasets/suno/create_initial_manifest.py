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
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Dict, Tuple

import jiwer
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


class CalculateUDWERParallel(BaseParallelProcessor):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @staticmethod
    def calculate_for_perm(permutation, reference):
        comb_error = 0

        for r_, h_ in zip(reference, permutation):
            measures = jiwer.compute_measures(r_, h_)
            errors = measures['insertions'] + measures['deletions'] + measures['substitutions']
            comb_error += errors

        return comb_error

    def get_ud_wer(self, r, h, ref_errors=None):
        r_utt_s = r.split(' <cs> ')
        h_utt_s = h.split(' <cs> ')

        if len(r_utt_s) < len(h_utt_s):
            padding = ['_'] * (len(h_utt_s) - len(r_utt_s))
            r_utt_s.extend(padding)

        if len(h_utt_s) < len(r_utt_s):
            padding = ['_'] * (len(r_utt_s) - len(h_utt_s))
            h_utt_s.extend(padding)

        assert len(r_utt_s) == len(h_utt_s)

        initial_error = 0
        num_words = 0

        for r_, h_ in zip(r_utt_s, h_utt_s):
            measures = jiwer.compute_measures(r_, h_)
            errors = measures['insertions'] + measures['deletions'] + measures['substitutions']

            initial_error += errors
            num_words += len(r_.split())

        if initial_error == ref_errors:
            wer = initial_error / num_words
            return wer, initial_error, num_words

        permutations = list(itertools.permutations(h_utt_s))[1:100]

        if len(permutations) > 1_000_000:
            return None, None, None

        print('GAGOOOOoooooooooo')

        print("LEN PERMUTATIONS: ", len(permutations))

        # k_errors = []

        # for perm in permutations:
        #     try:
        #         k_error = self.calculate_for_perm(perm, r_utt_s)
        #         k_errors.append(k_error)
        #     except Exception as e:
        #         print(e)
        #         print(r_utt_s)
        #         raise e

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(self.calculate_for_perm, permutations, repeat(r_utt_s)))

        # k_errors = itertools.chain(
        #             *process_map(
        #                 self.calculate_for_perm,
        #                 permutations,
        #                 [r_utt_s]*len(permutations),
        #                 max_workers=8,
        #                 chunksize=5,
        #             )
        #         )

        min_error = min(results)
        # print('MIN ERROR: ', min_error)
        # print('NUM WORDS: ', num_words)
        wer = min_error / num_words

        return wer, min_error, num_words

        # # Create pairs by combining each permutation of list2 with list1
        # min_error = 100000
        # words = 0

        # print('='*20)

        # print("REFERENCE: ", r)
        # print("PREDICTED: ", h)

        # print('NUMBER OF PERMUTATIONS: ', len(permutations))

        # for perm in tqdm(permutations):

        #     comb_error = 0
        #     comb_words = 0

        #     for r_, h_ in zip(r_utt_s, perm):

        #         measures = jiwer.compute_measures(r_, h_)
        #         errors = measures['insertions'] + measures['deletions'] + measures['substitutions']

        #         comb_error += errors
        #         comb_words += len(r_.split())

        #     wer = comb_error / comb_words

        #     if comb_error < min_error:
        #         min_error = comb_error

        #     if ref_errors and min_error == ref_errors:
        #         break

        # wer = min_error / words

        # print('WER: ', wer)

        # return wer, min_error, words

    @staticmethod
    def get_wer_without_cs(r, h):
        r = r.replace(' <cs>', '')
        h = h.replace(' <cs>', '')
        measures = jiwer.compute_measures(r, h)
        errors = measures['insertions'] + measures['deletions'] + measures['substitutions']

        wer = errors / len(r.split())

        return wer, errors, len(r.split())

    def process_dataset_entry(self, data_entry: str):
        r = data_entry['text']
        h = data_entry['pred_text']

        wer, error, word = self.get_wer_without_cs(r, h)
        ud_wer, ud_error, ud_word = self.get_ud_wer(r, h, ref_errors=error)

        data_entry['wer'] = wer
        data_entry['ud_wer'] = ud_wer

        data_entry['ud_error'] = ud_error
        data_entry['ud_word'] = ud_word

        data_entry['error'] = error
        data_entry['word'] = word

        # print(wer, ud_wer)
        # print('='*20)

        return [DataEntry(data=data_entry)]


class CalculateUDWER(BaseProcessor):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @staticmethod
    def calculate_for_perm(permutation, reference):
        comb_error = 0

        for r_, h_ in zip(reference, permutation):
            measures = jiwer.compute_measures(r_, h_)
            errors = measures['insertions'] + measures['deletions'] + measures['substitutions']
            comb_error += errors

        return [comb_error]

    @staticmethod
    def get_wer_without_cs(r, h):
        r = r.replace(' <cs>', '')
        h = h.replace(' <cs>', '')
        measures = jiwer.compute_measures(r, h)
        errors = measures['insertions'] + measures['deletions'] + measures['substitutions']

        wer = errors / len(r.split())

        return wer, errors, len(r.split())

    @staticmethod
    def get_ud_wer(r, h, ref_errors=None):
        r_utt_s = r.split(' <cs> ')
        h_utt_s = h.split(' <cs> ')

        if len(r_utt_s) < len(h_utt_s):
            padding = ['_'] * (len(h_utt_s) - len(r_utt_s))
            r_utt_s.extend(padding)

        if len(h_utt_s) < len(r_utt_s):
            padding = ['_'] * (len(r_utt_s) - len(h_utt_s))
            h_utt_s.extend(padding)

        assert len(r_utt_s) == len(h_utt_s)

        initial_error = 0
        num_words = 0

        for r_, h_ in zip(r_utt_s, h_utt_s):
            measures = jiwer.compute_measures(r_, h_)
            errors = measures['insertions'] + measures['deletions'] + measures['substitutions']

            initial_error += errors
            num_words += len(r_.split())

        if initial_error == ref_errors:
            wer = initial_error / num_words
            return wer, initial_error, num_words

        if len(h_utt_s) > 10:
            return None, None, None

        permutations = itertools.permutations(h_utt_s)

        if len(h_utt_s) < 5:
            k_errors = []

            for perm in permutations:
                try:
                    k_error = CalculateUDWER.calculate_for_perm(perm, r_utt_s)
                    k_errors.extend(k_error)
                except Exception as e:
                    print(e)
                    print(r_utt_s)
                    raise e

        else:
            # with ThreadPoolExecutor(max_workers=8) as executor:
            #     results = list(executor.map(self.calculate_for_perm, permutations, repeat(r_utt_s)))

            k_errors = itertools.chain(
                *process_map(
                    CalculateUDWER.calculate_for_perm,
                    permutations,
                    repeat(r_utt_s),
                    max_workers=16,
                    chunksize=10,
                )
            )

        min_error = min(k_errors)
        # print('MIN ERROR: ', min_error)
        # print('NUM WORDS: ', num_words)
        wer = min_error / num_words

        return wer, min_error, num_words

    def process(self):
        with open(self.input_manifest_file, "rt", encoding="utf8") as fin:
            # with open(self.output_manifest_file, 'wt', encoding='utf8') as fout:
            data_entries = []
            for line in tqdm(fin):
                data_entry = json.loads(line)

                r = data_entry['text']
                h = data_entry['pred_text']

                wer, error, word = self.get_wer_without_cs(r, h)
                ud_wer, ud_error, ud_word = self.get_ud_wer(r, h, ref_errors=error)

                data_entry['wer'] = wer
                # data_entry['ud_wer'] = ud_wer

                # data_entry['ud_error'] = ud_error
                # data_entry['ud_word'] = ud_word

                data_entry['error'] = error
                data_entry['word'] = word

                data_entries.append(data_entry)

        with open(self.output_manifest_file, 'wt', encoding='utf8') as fout:
            for data_entry in data_entries:
                json.dump(data_entry, fout, ensure_ascii=False)
                fout.write('\n')
                # fout.flush()
