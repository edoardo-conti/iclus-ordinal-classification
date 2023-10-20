"""Import modules."""
import os
import pickle
import random
from collections import defaultdict
import h5py
import tensorflow as tf
from keras.utils import Sequence, to_categorical
from tqdm import tqdm
import albumentations as A

class RichHDF5Dataset(Sequence):
    def __init__(self, file_path, pkl_frame_idxmap_path):
        self.file_path = file_path
        self.pkl_file_path = pkl_frame_idxmap_path
        self.h5file = h5py.File(file_path, 'r')
        self.group_names = list(self.h5file.keys())
        self.total_videos, self.total_frames, self.frame_index_map = self.elaborate_frameidx_map()

    def load_cached_data(self):
        if os.path.exists(self.pkl_file_path):
            with open(self.pkl_file_path, 'rb') as pickle_file:
                return pickle.load(pickle_file)
        else:
            return None

    def save_cached_data(self, data):
        with open(self.pkl_file_path, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

    def elaborate_frameidx_map(self):
        # Try to load cached data
        cached_data = self.load_cached_data()
        if cached_data is not None:
            total_videos, total_frames, frame_index_map = cached_data
            print(f"{total_videos} videos ({total_frames} frames) loaded from cached data.")
        else:
            total_videos = 0
            max_frame_idx_end = 0
            frame_index_map = {}

            # Create tqdm progress bar
            with tqdm(desc="Elaborating frames and index mapping", unit=' video', dynamic_ncols=True) as pbar:
                for group_name in self.group_names:
                    for video_name in self.h5file[group_name]:
                        video_group = self.h5file[group_name][video_name]
                        frame_idx_start = video_group.attrs['frame_idx_start']
                        frame_idx_end = video_group.attrs['frame_idx_end']
                        max_frame_idx_end = max(max_frame_idx_end, frame_idx_end)
                        for i in range(frame_idx_start, frame_idx_end + 1):
                            frame_index_map[i] = (group_name, video_name)
                        total_videos += 1
                        pbar.update(1)

            total_frames = max_frame_idx_end + 1

            # Save data to pickle file for future use
            self.save_cached_data((total_videos, total_frames, frame_index_map))
            print(f"\n{total_videos} videos ({total_frames} frames) loaded and cached.")

        return total_videos, total_frames, frame_index_map

    def __len__(self):
        return self.total_frames

    def __getitem__(self, index):
        if index < 0 or index >= self.total_frames:
            raise IndexError("Index out of range")

        group_name, video_name = self.frame_index_map[index]
        video_group = self.h5file[group_name][video_name]
        frame_data = video_group['frames'][f'frame_{index}'][:]
        target_data = video_group['targets'][f'target_{index}']

        # get metadata
        patient = video_group.attrs['patient']
        medical_center = video_group.attrs['medical_center']

        return (index, frame_data, target_data, patient, medical_center)


class HDF5Dataset(Sequence):
    def __init__(self, hdf5_dataset, indexes, batch_size=32, augmentation=False):
        self.hdf5_dataset = hdf5_dataset
        self.dataset_indexes = indexes
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.resize_size = (224, 224)
    
    def __len__(self):
        num_samples = len(self.dataset_indexes)
        num_batches = num_samples // self.batch_size
        if num_samples % self.batch_size != 0:
            num_batches += 1

        return num_batches
    
    def dataAugmentation(self, frame):
        min_size = min(frame.shape[0], frame.shape[1])
        transform = A.Compose([
            # I. Elastic Warping 
            A.ElasticTransform(alpha=150, sigma=10, alpha_affine=0.1, p=0.3),
            # II. Cropping
            A.RandomSizedCrop(min_max_height=(int(min_size * 0.7), min_size),
                              height=frame.shape[0], width=frame.shape[1], p=0.3),
            # III. Blurring
            A.GaussianBlur(blur_limit=(11, 21), p=0.3),
            # IV. Random rotation
            A.Rotate(limit=(-23, 23), p=0.5),
            # V. Contrast distortion
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
        ])

        return transform(image=frame)['image']

    def __getitem__(self, batch_index):
        start_idx = batch_index * self.batch_size
        end_idx = (batch_index + 1) * self.batch_size

        frames = []
        labels = []

        for index in self.dataset_indexes[start_idx:end_idx]:
            if index < len(self.hdf5_dataset):
                _, frame_data, target_data, _, _ = self.hdf5_dataset[index]
                
                # Perform data augmentation on-the-fly is enabled
                # note: to be applied before converting the frame to tf tensor
                if self.augmentation: 
                    frame_data = self.dataAugmentation(frame_data)

                # Convert the frame data into a tensor
                frame_tensor = tf.convert_to_tensor(frame_data)

                # Normalize image
                frame_tensor = tf.cast(frame_tensor, tf.float32) / 255.0

                # Apply resize transformation (mandatory for both train and test sets)
                frame_tensor = tf.image.resize(frame_tensor, self.resize_size, antialias=True)

                # One-hot encode the targets to get the label
                label = tf.squeeze(target_data)
                label_ohe = to_categorical(label, num_classes=4)

                # Batches the frames
                frames.append(frame_tensor)
                labels.append(label_ohe)

        # stack the frame to output the expected shape (es if bs=8: (8, 224, 224, 3))
        frames_batch = tf.stack(frames, axis=0)
        labels_batch = tf.stack(labels, axis=0)

        return frames_batch, labels_batch


def _load_dsdata_pickle(dataset, pkl_file):
    # Check if the pickle file exists
    if pkl_file and os.path.exists(pkl_file):
        # If the pickle file exists, load the data from it
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            medical_center_patients = data['medical_center_patients']
            data_index = data['data_index']
            data_map_idxs_pcm = data['data_map_idxs_pcm']
            score_counts = data['score_counts']
            labels = data['labels']
    else:
        # If the pickle file doesn't exist, create the data
        medical_center_patients = defaultdict(set)
        data_index = {}
        data_map_idxs_pcm = defaultdict(list)
        score_counts = defaultdict(int)
        labels = []  # List to store target labels

        for index, (_, _, target_data, patient, medical_center) in enumerate(tqdm(dataset)):
            medical_center_patients[medical_center].add(patient)
            data_index[index] = (patient, medical_center)
            data_map_idxs_pcm[(patient, medical_center)].append(index)
            score_counts[int(target_data[()])] += 1
            labels.append(int(target_data[()]))
        
        # Save the data to a pickle file if pkl_file is provided
        if pkl_file:
            data = {
                'medical_center_patients': medical_center_patients,
                'data_index': data_index,
                'data_map_idxs_pcm': data_map_idxs_pcm,
                'score_counts': score_counts,
                'labels': labels
            }
            
            with open(pkl_file, 'wb') as f:
                pickle.dump(data, f)
    
    return medical_center_patients, data_index, data_map_idxs_pcm, score_counts, labels


def split_strategy(dataset, ratios=[0.6, 0.2, 0.2], pkl_file=None, rseed=0):
    # Set the random seed for repeatability
    random.seed(rseed)

    if len(ratios) == 2:
        train_ratio, _ = ratios
        val_ratio = 0.0
    elif len(ratios) == 3:
        train_ratio, val_ratio, _ = ratios
    else:
        raise ValueError("Ratios list must have 1, 2, or 3 values that sum to 1.0")
    
    # 0. Gather the metadata
    medical_center_patients, data_index, data_map_idxs_pcm, score_counts, labels = _load_dsdata_pickle(dataset, pkl_file)

    # 1. Calculate the number of patients and frames for each medical center
    frames_by_center = defaultdict(int)
    frames_by_center_patient = defaultdict(lambda: defaultdict(int))

    for (patient, center) in data_index.values():
        frames_by_center[center] += 1
        frames_by_center_patient[center][patient] += 1
    
    # 2. Calculate the target number of frames for each split
    total_frames = sum(frames_by_center.values())
    train_frames = int(total_frames * train_ratio)
    val_frames = int(total_frames * val_ratio)
    test_frames = total_frames - train_frames - val_frames

    # 3. Create a dictionary to track patient percentages for each center
    patient_perc_by_center = defaultdict(lambda: defaultdict(float))
    for center, patients in medical_center_patients.items():
        patients = list(patients)

        for patient in patients:
            patient_frames = frames_by_center_patient[center][patient]
            patient_percentage = patient_frames / total_frames
            patient_perc_by_center[center][patient] = patient_percentage
    
    # 4. Splitting the dataset by patients taking into account frames ratio
    # lists
    train_indices = []
    val_indices = []
    test_indices = []

    # sets to store statistics about medical centers and patients
    train_patients_by_center = defaultdict(set)
    val_patients_by_center = defaultdict(set)
    test_patients_by_center = defaultdict(set)

    # 4.1 Test set
    while len(test_indices) < test_frames:
        center = random.choice(list(patient_perc_by_center.keys()))
        patients = list(patient_perc_by_center[center].keys())
        if patients:
            patient = random.choice(patients)
            if center in patient_perc_by_center and patient in patient_perc_by_center[center]:
                if len(test_indices) + patient_perc_by_center[center][patient] * total_frames <= test_frames:
                    test_indices.extend(data_map_idxs_pcm[(patient, center)])
                    test_patients_by_center[center].add(patient)
                    del patient_perc_by_center[center][patient]
                else:
                    # Se supera test_frames, cerca i pazienti rimasti che possono essere aggiunti per avvicinare il rapporto
                    remaining_frames = test_frames - len(test_indices)
                    candidates = [p for p in patients if patient_perc_by_center[center][p] * total_frames <= remaining_frames]
                    if candidates:
                        # Ordina i candidati in base a quanto si avvicinano al rapporto desiderato
                        candidates = sorted(candidates, key=lambda p: abs((len(test_indices) + patient_perc_by_center[center][p] * total_frames) / test_frames - 1))
                        
                        for best_candidate in candidates:
                            if len(test_indices) + patient_perc_by_center[center][best_candidate] * total_frames <= test_frames:
                                test_indices.extend(data_map_idxs_pcm[(best_candidate, center)])
                                test_patients_by_center[center].add(best_candidate)
                                del patient_perc_by_center[center][best_candidate]
                    else:
                        break

    # 4.2 Validation set
    while len(val_indices) < val_frames:
        center = random.choice(list(patient_perc_by_center.keys()))
        patients = list(patient_perc_by_center[center].keys())
        if patients:
            patient = random.choice(patients)
            if center in patient_perc_by_center and patient in patient_perc_by_center[center]:
                if len(val_indices) + patient_perc_by_center[center][patient] * total_frames <= val_frames:
                    val_indices.extend(data_map_idxs_pcm[(patient, center)])
                    val_patients_by_center[center].add(patient)
                    del patient_perc_by_center[center][patient]
                else:
                    # Se supera train_frames, cerca i pazienti rimasti che possono essere aggiunti per avvicinare il rapporto
                    remaining_frames = val_frames - len(val_indices)
                    candidates = [p for p in patients if patient_perc_by_center[center][p] * total_frames <= remaining_frames]
                    if candidates:
                        # Ordina i candidati in base a quanto si avvicinano al rapporto desiderato
                        candidates = sorted(candidates, key=lambda p: abs((len(val_indices) + patient_perc_by_center[center][p] * total_frames) / val_frames - 1))
                        
                        for best_candidate in candidates:
                            if len(val_indices) + patient_perc_by_center[center][best_candidate] * total_frames <= val_frames:
                                val_indices.extend(data_map_idxs_pcm[(best_candidate, center)])
                                val_patients_by_center[center].add(best_candidate)
                                del patient_perc_by_center[center][best_candidate]
                    else:
                        break
    
    # 4.3 Train set
    for center in patient_perc_by_center:
        for patient in patient_perc_by_center[center]:
            train_indices.extend(data_map_idxs_pcm[(patient, center)])
            train_patients_by_center[center].add(patient)
    
    # 5. Diagnostic checks and return values
    total_frames_calc = len(train_indices) + len(val_indices) + len(test_indices)
    if total_frames != total_frames_calc:
        print(f"dataset splitting gone wrong (expected: {total_frames}, got:{total_frames_calc})")
    
    # Sum up statistics info
    split_info = {
        'medical_center_patients': medical_center_patients,
        'frames_by_center': frames_by_center,
        'train_patients_by_center': train_patients_by_center,
        'val_patients_by_center': val_patients_by_center,
        'test_patients_by_center': test_patients_by_center,
        'frames_by_center_patient': frames_by_center_patient,
        'score_counts': score_counts,
        'labels': labels
    }

    train_idxs_p = round((len(train_indices) / len(dataset)) * 100)
    val_idxs_p = round((len(val_indices) / len(dataset)) * 100)
    test_idxs_p = 100 - (train_idxs_p + val_idxs_p)

    if val_ratio == 0.0:
        print(f"dataset split: train={len(train_indices)}({train_idxs_p}%), test={len(test_indices)}({test_idxs_p}%)")
        return train_indices, test_indices, split_info
    
    print(f"dataset split: train={len(train_indices)}({train_idxs_p}%), val={len(val_indices)}({val_idxs_p}%), test={len(test_indices)}({test_idxs_p}%)")

    return train_indices, val_indices, test_indices, split_info

    # Split the patients for each medical center
    train_indices = []
    val_indices = []
    test_indices = []

    # Lists to store statistics about medical centers and patients
    train_patients_by_center = defaultdict(set)
    valpatients_by_center = defaultdict(set)
    test_patients_by_center = defaultdict(set)
    frame_counts_by_center = defaultdict(int)
    frame_counts_by_center_patient = defaultdict(lambda: defaultdict(int))

    for medical_center, patients in medical_center_patients.items():
        patients = list(patients)
        random.shuffle(patients)
        train_index = int(train_ratio * len(patients))
        validation_index = int((train_ratio + val_ratio) * len(patients))

        for index, (patient, center) in data_index.items():
            if center == medical_center:
                if patient in patients[:train_index]:
                    train_indices.append(index)
                    train_patients_by_center[medical_center].add(patient)
                elif patient in patients[train_index:validation_index]:
                    val_indices.append(index)
                    valpatients_by_center[medical_center].add(patient)
                else:
                    test_indices.append(index)
                    test_patients_by_center[medical_center].add(patient)

                frame_counts_by_center[medical_center] += 1
                frame_counts_by_center_patient[medical_center][patient] += 1

    # Sum up statistics info
    split_info = {
        'medical_center_patients': medical_center_patients,
        'frame_counts_by_center': frame_counts_by_center,
        'train_patients_by_center': train_patients_by_center,
        'val_patients_by_center': valpatients_by_center,
        'test_patients_by_center': test_patients_by_center,
        'frame_counts_by_center_patient': frame_counts_by_center_patient,
        'total_train_frames': len(train_indices),
        'total_val_frames': len(val_indices),
        'total_test_frames': len(test_indices),
        'score_counts': score_counts,
        'labels': labels
    }

    train_idxs_p = round((len(train_indices) / len(dataset)) * 100)
    val_idxs_p = round((len(val_indices) / len(dataset)) * 100)
    test_idxs_p = round((len(test_indices) / len(dataset)) * 100)

    if val_ratio == 0.0:
        print(f"dataset split: train={len(train_indices)}({train_idxs_p}%), test={len(test_indices)}({test_idxs_p}%)")
        return train_indices, test_indices, split_info
    
    print(f"dataset split: train={len(train_indices)}({train_idxs_p}%), val={len(val_indices)}({val_idxs_p}%), test={len(test_indices)}({test_idxs_p}%)")

    return train_indices, val_indices, test_indices, split_info


def reduce_sets(train, val=[], test=[], perc=1.0):
    # Compute length of subsets
    num_train_samples = int(len(train) * perc)
    num_test_samples = int(len(test) * perc)

    # Create random subsets
    train_indices = random.sample(range(len(train)), num_train_samples)
    test_indices = random.sample(range(len(test)), num_test_samples)
    
    if val:
        num_val_samples = int(len(val) * perc)
        val_indices = random.sample(range(len(val)), num_val_samples)
        print(f"dataset reduction: {int(perc*100)}% (train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)})")
        return train_indices, val_indices, test_indices
    
    print(f"dataset reduction: {int(perc*100)}% (train={len(train_indices)}, test={len(test_indices)})")
    return train_indices, test_indices
    