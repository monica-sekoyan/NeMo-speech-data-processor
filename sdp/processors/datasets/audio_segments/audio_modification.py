from pathlib import Path
from moviepy.editor import concatenate_audioclips, AudioFileClip

def concatenate_audio_moviepy(audio_clip_paths, output_path):
    """Concatenates several audio files into one audio file using MoviePy
    and save it to `output_path`. Note that extension (mp3, etc.) must be added to `output_path`"""
    clips = [AudioFileClip(c) for c in audio_clip_paths]
    final_clip = concatenate_audioclips(clips)
    final_clip.write_audiofile(output_path, verbose=False, logger=None)

    for clip in clips:
        clip.close()


def construct_audio(filenames, 
                    durations,
                    parent_id,
                    min_duration,
                    audio_dir,
                    combined_audio_dir,
                    end=False):
    
    audios_to_combine = []

    curr_dur = 0

    if end:
        filenames = filenames[::-1]
        durations = durations[::-1]

    for filename, dur in zip(filenames, durations):
        if curr_dur < min_duration:
            curr_dur += dur 
            audios_to_combine.append(filename)
        else:
            break

    if end:
        audios_to_combine = audios_to_combine[::-1]

    full_paths = [Path(audio_dir, parent_id, audio).with_suffix('.opus').as_posix() for audio in audios_to_combine]

    output_dir = Path(combined_audio_dir, parent_id)
    output_dir.mkdir(exist_ok=True, parents=True)

    output_path = Path(output_dir, "_".join(audios_to_combine)).with_suffix('.wav')

    try:

        concatenate_audio_moviepy(full_paths, output_path)

    except Exception as e:
        print(e)

        return "corrupted", 0

    return output_path.as_posix(), curr_dur