import librosa
import numpy as np
import argparse
import noisereduce as nr

def load_audio(file_path):
    try:
        audio_signal, sample_rate = librosa.load(file_path, sr=None)
        return audio_signal, sample_rate
    except FileNotFoundError:
        raise FileNotFoundError("The audio file was not found. Check the provided file path.")
    except PermissionError:
        raise PermissionError("Permission denied for accessing the audio file.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading the audio file: {str(e)}")

def preprocess_audio(audio_signal, sample_rate):
    audio_signal, _ = librosa.effects.trim(audio_signal)
    audio_signal = nr.reduce_noise(y=audio_signal, sr=sample_rate)
    audio_signal = librosa.util.normalize(audio_signal)
    audio_signal = librosa.effects.preemphasis(audio_signal)
    return audio_signal

def detect_note_onsets_and_offsets(audio_signal, sample_rate):
    onset_env = librosa.onset.onset_strength(y=audio_signal, sr=sample_rate)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sample_rate, units='time')
    offsets = np.diff(onsets, append=len(audio_signal) / sample_rate)
    notes = []
    for i in range(len(onsets)):
        if i < len(onsets) - 1:
            end_time = onsets[i + 1]
        else:
            end_time = audio_signal.size / sample_rate
        notes.append((onsets[i], end_time))
    return notes

def transcribe_notes_with_librosa(audio_signal, sample_rate):
    harmonic, percussive = librosa.effects.hpss(audio_signal)
    pitches, magnitudes = librosa.piptrack(y=harmonic, sr=sample_rate)
    note_events = detect_note_onsets_and_offsets(harmonic, sample_rate)
    notes = []
    for start_time, end_time in note_events:
        frame_start = librosa.time_to_frames([start_time], sr=sample_rate)[0]
        frame_end = librosa.time_to_frames([end_time], sr=sample_rate)[0]
        if frame_end > frame_start:
            pitch_index = np.argmax(magnitudes[:, frame_start:frame_end].max(axis=1))
            pitch = pitches[pitch_index, frame_start:frame_end].max()
            if pitch > 0:
                note_name = librosa.hz_to_note(pitch)
                notes.append((note_name, round(start_time, 3), round(end_time, 3)))

    unique_notes = []

    for note in notes:
        if not unique_notes or unique_notes[-1][0] != note[0]:
            unique_notes.append(note)
        else:
            last_note = unique_notes.pop()
            combined_note = (last_note[0], last_note[1], round(note[2], 3))
            unique_notes.append(combined_note)
    return unique_notes

def enhanced_transcribe_notes(audio_signal, sample_rate):
    audio_signal = preprocess_audio(audio_signal, sample_rate)
    notes_librosa = transcribe_notes_with_librosa(audio_signal, sample_rate)
    return notes_librosa

def check_file_integrity(file_path):
    try:
        with open(file_path, 'rb') as file:
            file.read()
    except Exception as e:
        print(f"File cannot be read: {str(e)}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Process a single audio file to transcribe guitar notes.")
    parser.add_argument("file_path", help="Path to the audio file.")
    args = parser.parse_args()

    if not check_file_integrity(args.file_path):
        print("File is corrupted or unreadable.")
        return

    try:
        audio_signal, sample_rate = load_audio(args.file_path)
        notes = enhanced_transcribe_notes(audio_signal, sample_rate)
        print(notes)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
