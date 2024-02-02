import os
import random
import hashlib
import numpy as np
import pickle 
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.utils import check_random_state
from sklearn.utils.class_weight import compute_class_weight
from network.augmentation import USDataAugmentation

class DatasetHandler():
    def __init__(self, 
                 dataset_dir:str,
                 input_size:int = 224, 
                 num_classes:int = 4, 
                 shuffle_bsize:int = 100, 
                 seed:int = 42):
            self.dataset_dir = dataset_dir
            self.input_size = input_size
            self.num_classes = num_classes
            self.shuffle_bsize = shuffle_bsize
            self.seed = seed
            
            # initial mapping of the patient's movies
            self.mov_per_pat = {}
            
            # stratified group k-fold
            self.movies = []
            self.groups = []
            self.labels = []
            self.folds = None

            # TFRecord dataset descriptor             
            self.feature_description = {
                'frame': tf.io.FixedLenFeature([], tf.string),
                'score': tf.io.FixedLenFeature([], tf.int64)
            }

            # data augmentation handler
            self.augmenter = USDataAugmentation(input_size=input_size, seed=seed)
            
            # TODO: debugging
            self.labels_pickle = 'kfold_labels.pkl' 


    # class building
    def build(self):
        self.mov_per_pat = self.map_movies_per_patient()
        self.augmenter.build()


    # map movies for each patient of the dataset
    def map_movies_per_patient(self):
        movies_per_patient = {}

        for medical_center in os.listdir(self.dataset_dir):
            medical_center_folder = os.path.join(self.dataset_dir, medical_center)

            if os.path.isdir(medical_center_folder):
                for patient in os.listdir(medical_center_folder):
                    patient_folder = os.path.join(medical_center_folder, patient)

                    if os.path.isdir(patient_folder):
                        movies = os.listdir(patient_folder)
                        tfrecord_files = [os.path.join(patient_folder, movie) for movie in movies if movie.endswith('.tfrecord')]

                        unique_patient_key = f'{medical_center}/{patient}'
                        movies_per_patient[unique_patient_key] = tfrecord_files
        
        return movies_per_patient


    # counting the number of frames at two different levels (per-patient and per-center)
    def count_frames_per_patient_and_center(self):
        frame_count_per_patient = {}
        frame_count_per_center = {}

        for patient_key, tfrecord_files in self.mov_per_pat.items():
            dataset = tf.data.TFRecordDataset(tfrecord_files)
            frame_count_per_patient[patient_key] = sum(1 for _ in dataset.as_numpy_iterator())

        for medcenter_pat, frames in frame_count_per_patient.items():
            med_center = medcenter_pat.split('/')[0]
            frame_count_per_center[med_center] = frame_count_per_center.get(med_center, 0) + frames

        return frame_count_per_patient, frame_count_per_center


    # TODO: debugging
    def save_labels_to_file(self):
        with open(self.labels_pickle, 'wb') as f:
            pickle.dump(self.labels, f)

    def load_labels_from_file(self):
        if os.path.exists(self.labels_pickle):
            with open(self.labels_pickle, 'rb') as f:
                self.labels = pickle.load(f)
            return True
        else:
            return False
    
    # Stratified Group K-Folding 
    def sgkfold(self, num_folds=3, shuffle_folds=True):
        self.movies = [movie_path for movie_paths in self.mov_per_pat.values() for movie_path in movie_paths]
        self.groups = [patient for patient, patient_movie in self.mov_per_pat.items() for _ in range(len(patient_movie))]
        
        # TODO: load labels from file if available
        if not self.load_labels_from_file():
            for movie in self.movies:
                movie_ds = tf.data.TFRecordDataset(movie)
                movie_labels = self.extract_labels_from_tfrset(movie_ds)
                movie_majority_label = np.argmax(np.bincount(movie_labels, minlength=self.num_classes))
                self.labels.append(movie_majority_label)
            
            self.save_labels_to_file()  # Save labels to file
        
        random_state = check_random_state(self.seed)
        sgkf = StratifiedGroupKFold(n_splits=num_folds, shuffle=shuffle_folds, random_state=random_state)
        self.folds = sgkf.split(self.movies, self.labels, self.groups)

        return sgkf.get_n_splits(self.movies, self.labels)


    # stratified random holdout splits
    def n_strat_shuffle_split(self, patients, labels, val_ratio=0.15, splits=3, state=None):                
        seed = int(hashlib.sha256(f'{self.seed}_{state}'.encode('utf-8')).hexdigest(), 16) % (2**32)
        random_state = check_random_state(seed)
        
        sss = StratifiedShuffleSplit(n_splits=splits, test_size=val_ratio, random_state=random_state)
        
        return sss.split(patients, labels)


    # method to sample a score for each patient based on per-video scores
    # in order to being able to stratify per-patient splits 
    def sample_score_per_patient(self, train_groups, train_labels):
        # create a numpy array where each row contains the label associated with a patient
        labels_in_column = np.column_stack((train_groups, train_labels))

        # create a dictionary where the key is the patient, and the value is a list of labels
        patient_dictionary = {}
        for row in labels_in_column:
            patient = row[0]
            label = row[1]
            if patient not in patient_dictionary:
                patient_dictionary[patient] = [label]
            else:
                patient_dictionary[patient].append(label)

        # convert the dictionary into a list of lists
        patients_labels = list(patient_dictionary.values())
        
        # apply class weights to each sublist using bincount and argmax
        class_weight = compute_class_weight('balanced', classes=np.unique(self.labels), y=self.labels)
        result_list = [np.argmax(np.multiply(np.bincount(sublist, minlength=4), class_weight)) for sublist in patients_labels]

        # perform additional check and replacement for the least populated class
        result_list_bincount = np.bincount(result_list, minlength=4)
        if result_list_bincount[-1] == 1:
            least_populated_class = np.argmin(result_list_bincount)
            near_least_populated_class = least_populated_class - 1

            indices_to_replace = [i for i, x in enumerate(result_list) if x == near_least_populated_class]
            
            if indices_to_replace:
                random.seed(self.seed)
                index_to_replace = random.choice(indices_to_replace)
                result_list[index_to_replace] = least_populated_class
        
        return result_list
    

    # extract set's labels parsing only the scores to avoid computing the frames
    def extract_labels_from_tfrset(self, dataset):
        def _extract_label(example_proto):
            return tf.io.parse_single_example(example_proto, self.feature_description)['score']
        
        labels = list(dataset.map(_extract_label).as_numpy_iterator())

        return np.array(labels)


    # create tfrecord dataset from patients keys    
    def build_tfrecord_from_patients(self, patient_keys, extract_labels=True):
        if type(patient_keys) is not list:
            patient_keys = [patient_keys]
        
        movies_per_patients = [self.mov_per_pat[patient] for patient in patient_keys]
        tfrecord_files = [movie for patient_movies in movies_per_patients for movie in patient_movies]

        dataset = tf.data.TFRecordDataset(tfrecord_files)
        labels = self.extract_labels_from_tfrset(dataset) if extract_labels else None

        return dataset, labels
    

    # generate the sets using TFRecordDataset
    def prepare_tfrset(self, split_set, random_under_msampler=False):
        dataset, labels = self.build_tfrecord_from_patients(self.split[split_set], random_under_msampler)
        
        return dataset, labels


    # function to parse LUS video to get frames and labels
    def _parse_lus_movie(self, example_proto):
        record = tf.io.parse_single_example(example_proto, self.feature_description)
        
        # frame
        frame_data = tf.io.decode_jpeg(record['frame'], channels=3)
        frame = tf.image.resize(frame_data, [self.input_size, self.input_size]) / 255.0

        # score
        label = record['score']
        label = tf.one_hot(label, self.num_classes)
            
        return frame, label


    # set generator to be fed into neural network 
    def generate_tfrset(self, pre_dataset, batch_size, shuffle=False, augment=False):
        # mapping
        dataset = pre_dataset.map(self._parse_lus_movie, num_parallel_calls=tf.data.AUTOTUNE)
        
        # shuffling
        if shuffle:
            self.shuffle_bsize /= 2 if batch_size > 32 else 1
            computed_buffer_size = batch_size * int(self.shuffle_bsize)
            dataset = dataset.shuffle(buffer_size=computed_buffer_size, reshuffle_each_iteration=True)
        
        # batching
        dataset = dataset.batch(batch_size)

        # data augmentation
        if augment:
            dataset = dataset.map(lambda x, y: (self.augmenter.us_augmentation(x), y), num_parallel_calls=batch_size)
        
        # infinite and prefetching
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset
    

    # util function to plot batches of specific dataset
    def plot_set_batches(self, set, batch_size, num_batches=10):
        # color map
        class_colors = {0: 'green', 1: 'darkblue', 2: 'darkorange', 3: 'darkred'}

        for batch in set.take(num_batches): 
            _, axes = plt.subplots(batch_size // 8, 8, figsize=(20, 3 * (batch_size // 8)))
            
            frames, labels = batch
            
            for i, (frame, label) in enumerate(zip(frames, labels)):        
                # print the image in the grid
                axes[i // 8, i % 8].imshow(frame)

                # set the label color
                color = class_colors.get(np.argmax(label), 'black')
                axes[i // 8, i % 8].set_title(f'Target: {label}', color=color)
                
                # hide the axis
                axes[i // 8, i % 8].axis('off')

            plt.tight_layout()
            plt.show()
            plt.close()