"""Import modules."""
import os
import pickle
import random
from collections import defaultdict
import h5py
import tensorflow as tf
from keras.utils import Sequence
from tqdm import tqdm
import albumentations as A
import cv2
import keras_cv
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

class HDF5Dataset(Sequence):
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

        return total_videos, total_frames, frame_index_map

    def __len__(self):
        return self.total_frames

    def __getitem__(self, index):
        if index < 0 or index >= self.total_frames:
            raise IndexError("Index out of range")

        # get group and video
        group_name, video_name = self.frame_index_map[index]
        video_group = self.h5file[group_name][video_name]
        
        # get frames and labels
        frame_data = video_group['frames'][f'frame_{index}'][:]
        target_data = video_group['targets'][f'target_{index}']

        # get metadata
        patient = video_group.attrs['patient']
        medical_center = video_group.attrs['medical_center']

        return index, frame_data, target_data, patient, medical_center


def load_ds_metadata(dataset, pkl_file, only_labels=False):
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

    if only_labels:
        return np.array(labels)
    
    return medical_center_patients, data_index, data_map_idxs_pcm, score_counts, labels


class TFDataset():
    def __init__(self, hdf5_dataset, set_idxs, batch_size=16, buffer_size=100, is_train=False, augmentation=False):
        self.hdf5_dataset = hdf5_dataset
        self.set_idxs = set_idxs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.is_train = is_train
        self.augmentation = augmentation

        # params to be computed
        self.dataset = None
    
    def label_smooth(self, labels, epsilon=0.1):
        labels = tf.cast(labels, tf.float32)
        num_classes = labels.shape[-1]
        return (1.0 - epsilon) * labels + epsilon / num_classes

    def data_augmentation(self, frame):
        min_size = min(frame.shape[0], frame.shape[1])
        zoom = 0.2

        transform = A.Compose([
            # I. Elastic Warping (verified: better to avoid)
            A.ElasticTransform(alpha=100, sigma=10, alpha_affine=0.1, p=0.0),
            # II. Cropping (verified: avoid big zoom values)
            A.RandomSizedCrop(min_max_height=(int(min_size * (1 - zoom)), min_size),
                            height=frame.shape[0], width=frame.shape[1], p=0.3),
            # III. Blurring (verified: not useful but with small limits it shouldn't be dangerous)
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            # IV. Random rotation (verified)
            A.Rotate(limit=(-23, 23), p=0.3),
            # V. Contrast distortion (verified)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.15, p=0.3),
            # VII. Horizontal Flip (verified, not found in benchmark's paper but in Roy et al.)
            A.HorizontalFlip(p=0.3)
        ])
        return transform(image=frame)['image']

    def mixup_batch(self, frames, labels):
        mixup = keras_cv.layers.MixUp()

        labels = self.label_smooth(labels)
        
        mixed_batch = mixup({'images': frames, 'labels': labels})

        return mixed_batch['images'], mixed_batch['labels']

    def _process_frames(self, frame_data, target_data):
        frame_tensor = tf.cast(frame_data, tf.float32) / 255.0
        
        label = tf.squeeze(target_data)
        label_ohe = tf.one_hot(label, depth=4, dtype=tf.int32)

        return frame_tensor, label_ohe

    def _map_frames(self, index):
        index = index.numpy()
        _, frame_data, target_data, _, _ = self.hdf5_dataset[index]
        
        frame_data = cv2.resize(frame_data, (224, 224))
        
        if self.is_train and self.augmentation is True:
            frame_data = self.data_augmentation(frame_data)

        return frame_data, target_data

    def create_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.set_idxs)
        
        if self.is_train:
            dataset = dataset.shuffle(buffer_size=self.batch_size * self.buffer_size, reshuffle_each_iteration=True)
        
        dataset = dataset.map(lambda index: tf.py_function(func=self._map_frames, inp=[index], Tout=(tf.float32, tf.int32)), 
                                num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda f, t: tf.py_function(func=self._process_frames, inp=[f, t], Tout=(tf.float32, tf.int32)),
                                num_parallel_calls=tf.data.AUTOTUNE)
        
        if self.is_train and self.augmentation == 'mixup':
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.map(lambda f, t: tf.py_function(func=self.mixup_batch, inp=[f, t], Tout=(tf.float32, tf.float32)), 
                                  num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.unbatch()

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        self.dataset = dataset

        return self.dataset

    def as_iterator(self):
        if self.dataset is None:
            self.create_dataset()
        
        return self.dataset.as_numpy_iterator()


def split_by_patients(dataset, ratio=None, th=0.95, pkl_file=None, rseed=42):
    # Set the random seed for repeatability
    random.seed(rseed)
    
    if ratio is None:
        ratio = [0.6, 0.2, 0.2]

    if len(ratio) == 2:
        train_ratio, _ = ratio
        val_ratio = 0.0
    elif len(ratio) == 3:
        train_ratio, val_ratio, _ = ratio
    else:
        raise ValueError("ratio list must have 1, 2, or 3 values that sum to 1.0")
    
    # 0. gather the metadata
    medical_center_patients, _, data_map_idxs_pcm, score_counts, labels = load_ds_metadata(dataset, pkl_file)    

    # 1. calculate the target number of frames for each split
    total_frames = len(labels)
    train_frames = int(total_frames * train_ratio * (2.0 - th))
    val_frames = int(total_frames * val_ratio * th)
    test_frames = total_frames - train_frames - val_frames
    
    # 2. splitting the dataset by patients taking into account sets split ratio
    train_idxs = []
    val_idxs = []
    test_idxs = []

    # sets to store statistics about medical centers and patients
    train_patients_by_center = defaultdict(set)
    val_patients_by_center = defaultdict(set)
    test_patients_by_center = defaultdict(set)
    
    # 2.1 test set
    while (len(test_idxs) < test_frames):
        center = random.choice(list(medical_center_patients.keys()))
        patients = medical_center_patients[center]
        try:
            patient = patients.pop()
            test_idxs.extend(data_map_idxs_pcm[(patient, center)])
            test_patients_by_center[center].add(patient)
        except (KeyError, IndexError):
            del medical_center_patients[center]

        
    # 2.2 validation set
    while (len(val_idxs) < val_frames):
        center = random.choice(list(medical_center_patients.keys()))
        patients = medical_center_patients[center]
        try:
            patient = patients.pop()
            val_idxs.extend(data_map_idxs_pcm[(patient, center)])
            val_patients_by_center[center].add(patient)
        except (KeyError, IndexError):
            del medical_center_patients[center]
    
    # 2.3 training set
    for center in list(medical_center_patients.keys()):
        for patient in list(medical_center_patients[center]):
            train_idxs.extend(data_map_idxs_pcm[(patient, center)])
            train_patients_by_center[center].add(patient)
    
    # 4. compliance checks
    total_frames_calc = len(train_idxs) + len(val_idxs) + len(test_idxs)
    if total_frames != total_frames_calc:
        raise ValueError(f"splitting gone wrong (expected: {total_frames}, got:{total_frames_calc})")
    
    # 5. sum up statistics info
    split_info = {
        'medical_center_patients': medical_center_patients,
        'train_patients_by_center': train_patients_by_center,
        'val_patients_by_center': val_patients_by_center,
        'test_patients_by_center': test_patients_by_center,
        'score_counts': score_counts,
        'labels': labels
    }

    if val_ratio == 0.0:
        return train_idxs, test_idxs, split_info

    return train_idxs, val_idxs, test_idxs, split_info


def split_by_centers(dataset, ratio=None, pkl_file=None, rseed=42):
    # Set the random seed for repeatability
    random.seed(rseed)
    
    if ratios is None:
        ratios = [0.5, 0.5]

    if len(ratios) == 2:
        val_ratio, test_ratio = ratios
    else:
        raise ValueError("ratios list must have 2 values that sum to 1.0")

    # 0. gather the metadata
    medical_center_patients, _, data_map_idxs_pcm, score_counts, labels = load_dsdata_pickle(dataset, pkl_file)    
    # 0.1 setup the returning split infos
    split_info = {'medical_center_patients': medical_center_patients.copy()}

    # 1. calculate the target number of frames for each split
    total_frames = len(labels)
    
    # 2. splitting the dataset by medical centers taking into account sets split ratio
    train_idxs = []
    val_idxs = []
    test_idxs = []

    # sets to store statistics about medical centers and patients
    train_patients_by_center = defaultdict(set)
    val_patients_by_center = defaultdict(set)
    test_patients_by_center = defaultdict(set)
    
    # 2.1 training set
    brescia_patients = medical_center_patients.pop('Brescia', set())
    for patient in brescia_patients:
        train_idxs.extend(data_map_idxs_pcm[(patient, 'Brescia')])
        train_patients_by_center['Brescia'].add(patient)

    train_frames = len(train_idxs)
    test_frames = (total_frames - train_frames) * test_ratio

    # 2.2 test set
    while len(test_idxs) < test_frames:
        if not medical_center_patients:
            break

        centers = list(medical_center_patients.keys())
        center = random.choice(centers)

        patients = list(medical_center_patients[center])
        for patient in patients:
            test_idxs.extend(data_map_idxs_pcm[(patient, center)])
            test_patients_by_center[center].add(patient)
        del medical_center_patients[center]

    # 2.3 validation set
    for center in list(medical_center_patients.keys()):
        for patient in list(medical_center_patients[center]):
            val_idxs.extend(data_map_idxs_pcm[(patient, center)])
            val_patients_by_center[center].add(patient)
    
    # 4. compliance checks
    total_frames_calc = len(train_idxs) + len(val_idxs) + len(test_idxs)
    if total_frames != total_frames_calc:
        raise ValueError(f"splitting gone wrong (expected: {total_frames}, got:{total_frames_calc})")
    
    # 5. sum up statistics info
    additional_info = {
        'train_patients_by_center': train_patients_by_center,
        'val_patients_by_center': val_patients_by_center,
        'test_patients_by_center': test_patients_by_center,
        'score_counts': score_counts,
        'labels': labels
    }
    split_info.update(additional_info)

    if val_ratio == 0.0:
        return train_idxs, test_idxs, split_info

    return train_idxs, val_idxs, test_idxs, split_info


def rus(x_train, y_train, strategy='auto', rseed=42):   
    if 'rus_c' in strategy:
        strategy_class_index = int(strategy[-1]) - 1
        sorted_class_indices = np.argsort(np.bincount(y_train))
        minority_class = sorted_class_indices[0]
        target_class = sorted_class_indices[strategy_class_index]
        desired_count = np.sum(y_train == target_class)
        strategy = {key: desired_count for key in range(4) if key not in (minority_class, target_class)}
    else:
        strategy = 'auto'
    
    X_resampled = []
    y_resampled = []
    try:
        rus = RandomUnderSampler(sampling_strategy=strategy, random_state=rseed)
        X_resampled, y_resampled = rus.fit_resample(np.array(x_train).reshape(-1, 1), y_train)
    except ValueError:
        # TODO:
        raise ValueError('cannot undersample with the class provided.')
    
    X_resampled = np.array(X_resampled).reshape(-1).tolist()
    y_resampled = np.array(y_resampled).reshape(-1).tolist()

    return X_resampled, y_resampled
    

def trim_sets(train, val=[], test=[], perc=1.0):
    # Compute length of subsets
    num_train_samples = int(len(train) * perc)
    num_test_samples = int(len(test) * perc)

    # Create random subsets
    train_indices = random.sample(range(len(train)), num_train_samples)
    test_indices = random.sample(range(len(test)), num_test_samples)
    
    if val:
        num_val_samples = int(len(val) * perc)
        val_indices = random.sample(range(len(val)), num_val_samples)
        
        return train_indices, val_indices, test_indices
    
    return train_indices, test_indices
    