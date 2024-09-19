import os
import sys
import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_audio(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        return audio, sample_rate
    except FileNotFoundError:
        raise FileNotFoundError("The audio file was not found. Check the provided file path.")
    except PermissionError:
        raise PermissionError("Permission denied for accessing the audio file.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading the audio file: {str(e)}")

def extract_features(file_path):
    audio, sample_rate = load_audio(file_path)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mean_mfcc = np.mean(mfcc, axis=1)
    return mean_mfcc


def cluster_audios(folder_path):
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]
    features = np.array([extract_features(fp) for fp in file_paths])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    pca = PCA(n_components=3)
    reduced_features = pca.fit_transform(scaled_features)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(reduced_features)
    labels = kmeans.labels_
    clusters = {i: [] for i in range(2)}
    for label, file_path in zip(labels, file_paths):
        clusters[label].append(os.path.basename(file_path))
    return [tuple(cluster) for cluster in clusters.values()]


def main(folder_path):
    clusters = cluster_audios(folder_path)
    for cluster in clusters:
        print(cluster)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python audio_task_2.py path_to_audio_folder")
        sys.exit(1)
    folder_path = sys.argv[1]
    main(folder_path)

