import os
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from network.augmentation import USDataAugmentation

class DatasetHandler():
    def __init__(self, 
                 dataset_dir, 
                 input_size=224, 
                 num_classes=4, 
                 shuffle_bsize=None, 
                 random_state=42):
            self.dataset_dir = dataset_dir
            self.input_size = input_size
            self.num_classes = num_classes
            self.shuffle_bsize = shuffle_bsize
            self.random_state = random_state

            self.ds_mapping = {}
            self.split = {}
            self.frame_counts = {}
            
            self.feature_description = {
                'frame': tf.io.FixedLenFeature([], tf.string),
                'score': tf.io.FixedLenFeature([], tf.int64)
            }

            self.augmenter = USDataAugmentation(input_size=input_size, random_state=random_state)
            

    # build della classe
    def build(self):
        random.seed(self.random_state)
        self.ds_mapping = self.map_movies_per_patient()
        self.augmenter.build()


    # conteggio numero di frame per ogni paziente (aggregazione dei tfrecords corrispondenti)
    def map_movies_per_patient(self):
        ds_mapping = {}

        for medical_center in os.listdir(self.dataset_dir):
            medical_center_folder = os.path.join(self.dataset_dir, medical_center)

            if os.path.isdir(medical_center_folder):
                for patient in os.listdir(medical_center_folder):
                    patient_folder = os.path.join(medical_center_folder, patient)
    
                    if os.path.isdir(patient_folder):
                        movies = os.listdir(patient_folder)
                        tfrecord_files = [os.path.join(patient_folder, movie) for movie in movies if movie.endswith('.tfrecord')]

                        key = f'{medical_center}/{patient}'
                        ds_mapping[key] = tfrecord_files
        
        return ds_mapping


    # dataset splitting based on random patients's selection
    def split_dataset(self, split_ratio=[0.6, 0.2, 0.2]):
        train_ratio, test_ratio, val_ratio = split_ratio
        keys = list(self.ds_mapping.keys())
        
        # Effettua lo splitting delle chiavi
        train_keys, test_val_keys = train_test_split(keys, train_size=train_ratio, random_state=self.random_state)
        val_keys, test_keys = train_test_split(test_val_keys, test_size=test_ratio/(val_ratio+test_ratio), random_state=self.random_state)
        
        self.split = {'train': train_keys, 'val': val_keys, 'test': test_keys}


    # extract set's labels parsing only the scores to avoid computing the frames
    def extract_labels_from_tfrset(self, dataset):
        def _extract_label(example_proto):
            return tf.io.parse_single_example(example_proto, self.feature_description)['score']
        
        labels = list(dataset.map(_extract_label).as_numpy_iterator())

        return np.array(labels)


    # movies-based undersampling
    def movie_undersampler(self, movies_per_patients, factor=0.5):
        return [
            random.sample(patient_movies, max(1, int(len(patient_movies) * factor)))
            for patient_movies in movies_per_patients
        ]


    # generate the sets using TFRecordDataset
    def prepare_tfrset(self, split_set, random_under_msampler=False):
        #tfrecord_files = [movie for patient in self.split[split_set] for movie in self.ds_mapping[patient]]
        
        patients_per_set = self.split[split_set]
        # undersampling sui pazienti

        movies_per_patients = [self.ds_mapping[patient] for patient in patients_per_set]
        # undersampling sui video di ciascun paziente
        if random_under_msampler:
             movies_per_patients = self.movie_undersampler(movies_per_patients, random_under_msampler)
        
        tfrecord_files = [movie for patient_movies in movies_per_patients for movie in patient_movies]

        dataset = tf.data.TFRecordDataset(tfrecord_files)
        labels = self.extract_labels_from_tfrset(dataset)

        self.frame_counts[split_set] = len(labels)

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
            computed_buffer_size = batch_size * self.shuffle_bsize
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
