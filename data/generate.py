import os
import pyedflib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, resample

ORIGINAL_SAMPLE_RATE = 100
SAMPLE_RATE = 100       # Hz
WINDOW_DURATION = 30    # seconds
WINDOW_SIZE = SAMPLE_RATE * WINDOW_DURATION
TARGET_SIZE = (32, 32)


path_test_hypnogram = 'raw/test/hypnograms'
path_test_polysomnogram = 'raw/test/polysomnograms/'

path_train_hypnogram = 'raw/train/hypnograms'
path_train_polysomnogram = 'raw/train/polysomnograms/'

stage_mapping = {
  'Sleep stage W': 0,
  'Sleep stage 1': 1,
  'Sleep stage 2': 2,
  'Sleep stage 3': 3,
  'Sleep stage 4': 3,
  'Sleep stage R': 4
}

def resample_signal(signal, original_fs, target_fs):
  num_samples = int(len(signal) * float(target_fs) / original_fs)
  resampled_signal = resample(signal, num_samples)
  return resampled_signal

def remove_preceding_stage(signal, annotations):
  samples_to_remove = int(SAMPLE_RATE * annotations['duration'].iloc[0])
  new_signal = signal[samples_to_remove:]
  annotations_df = pd.DataFrame({
    'onset': [o - annotations['onset'].iloc[1] for o in annotations['onset'].iloc[1:]],
    'duration': annotations['duration'].iloc[1:],
    'stage': annotations['stage'].iloc[1:]
  })
  annotations_df.reset_index(drop=True, inplace=True)
  return new_signal, annotations_df

def remove_tailing_stage(signal, annotations):
  samples_to_remove = int(SAMPLE_RATE * annotations['duration'].iloc[-1])
  new_signal = signal[:-samples_to_remove]
  annotations_df = pd.DataFrame({
    'onset': annotations['onset'].iloc[:-1],
    'duration': annotations['duration'].iloc[:-1],
    'stage': annotations['stage'].iloc[:-1]
  })
  return new_signal, annotations_df

def generate_data(hyp_folder, psg_folder, is_train = True):
  spectrograms = []
  labels = []
  
  for hyp_filename in os.listdir(hyp_folder):
    hyp_path = os.path.join(hyp_folder, hyp_filename)

    if os.path.isfile(hyp_path):
      patient_identifier = hyp_filename[:8]
      patient_id = patient_identifier[:5]

      suffixes = ['E0', 'F0', 'G0', 'H0']
      psg_path = None
      for suffix in suffixes:
        psg_filename = f'{patient_identifier[:6]}{suffix}-PSG.edf'
        temp_path = os.path.join(psg_folder, psg_filename)
        if os.path.exists(temp_path):
          psg_path = temp_path
          break
      
      if psg_path is None:
        print(f"Warning: No matching PSG file found for {patient_identifier}")
        continue 
      try:
        p_signal = pyedflib.EdfReader(psg_path)
        raw_fpz_cz = p_signal.readSignal(0)
        p_signal.close()
      except Exception as e:
        print(f"Error reading PSG file for {patient_identifier}: {str(e)}")
        continue 

      f_hypnogram = pyedflib.EdfReader(hyp_path)
      annotations = f_hypnogram.readAnnotations()
      f_hypnogram._close()
      annotations_df = pd.DataFrame({
        'onset': annotations[0],
        'duration': annotations[1],
        'stage': annotations[2]
      })
      annotations_df['stage'] = annotations_df['stage'].map(stage_mapping)
      
      sig, ann = remove_preceding_stage(raw_fpz_cz, annotations_df)
      sig, ann = remove_tailing_stage(sig, ann)
      sig, ann = remove_tailing_stage(sig, ann)
      # sig = resample_signal(sig, ORIGINAL_SAMPLE_RATE, SAMPLE_RATE)
      
      for onset_time in range(0, int(ann['onset'].iloc[-1]), WINDOW_DURATION):
        img_filename = f'spectrogram_{patient_identifier[:6]}_{onset_time}.png'

        start_sample = onset_time * SAMPLE_RATE
        end_sample = start_sample + WINDOW_SIZE
        if end_sample > len(sig):
          break
        window = sig[start_sample:end_sample]
        
        window_onset = onset_time
        window_label = None
        for idx, onset in enumerate(ann['onset']):
          if onset <= window_onset < onset + ann['duration'][idx]:
              window_label = ann['stage'][idx]
              break
        labels.append([img_filename, window_label])

        px = 1 / plt.rcParams['figure.dpi']
        fig, ax = plt.subplots(figsize=(TARGET_SIZE[0]*px, TARGET_SIZE[1]*px))
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        NFFT = 128
        noverlap = int(NFFT * 0.3)      
        Pxx, freqs, bins, im = ax.specgram(window, NFFT=NFFT, Fs=SAMPLE_RATE, cmap='viridis', noverlap=noverlap)

        if is_train:
          image_path = os.path.join(f'processed/train/', img_filename)
        else:
          image_path = os.path.join(f'processed/test/', img_filename)

        fig.savefig(image_path, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Finished {img_filename}")

  labels_df = pd.DataFrame(labels, columns=['image_name', 'label'])
  labels_df = labels_df.dropna(subset=['label'])

  if is_train:
    class_counts = labels_df['label'].value_counts()
    max_count = class_counts.max()
    
    oversampled_data = []
    for label, count in class_counts.items():
      class_data = labels_df[labels_df['label'] == label]
      if count < max_count:
        oversampled = class_data.sample(n=max_count-count, replace=True, random_state=42)
        oversampled_data.append(oversampled)
      oversampled_data.append(class_data)
    
    oversampled_df = pd.concat(oversampled_data, ignore_index=True)
    
    oversampled_df.to_csv('labels_train.csv', index=False)
    labels_df.to_csv('non_oversampled_df.csv', index=False)
  else:
    labels_df.to_csv('labels_test.csv', index=False)

generate_data(path_train_hypnogram, path_train_polysomnogram, True)
generate_data(path_test_hypnogram, path_test_polysomnogram, False)