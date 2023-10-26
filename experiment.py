import json
import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn.utils.class_weight import compute_class_weight
from dataset import RichHDF5Dataset, HDF5Dataset, split_strategy, reduce_sets
from utilities import print_split_diagnostic_info, plot_fsplit_info, plot_psplit_info
from network import Net
from losses import make_cost_matrix, qwk_loss, ordinal_distance_loss
from metrics import np_quadratic_weighted_kappa, ccr, ms, mae, accuracy_oneoff, from_obdout_to_labels
from metrics_new import MyMetrics

class Experiment:
    def __init__(self, seed, exps_json, exp_idx):
        self.dataset_h5 = "/Users/edoardoconti/Tesi/iclus/dataset.h5"
        self.pkl_framesmap = "/Users/edoardoconti/Tesi/iclus/hdf5_frame_index_map.pkl"
        self.pkl_centersdict = "/Users/edoardoconti/Tesi/iclus/hospitals-patients-dict.pkl"
        self.seed = seed
        self.settings = self.load_json(exps_json)[exp_idx]

        # =========================
        # ======== dataset ========
        # =========================
        self.ds_img_size = 224
        self.ds_num_channels = 3
        self.ds_num_classes = 4
        self.dataset = None
        self.ds_infos = None
        self.train_class_weights = None
        # ======= train set =======
        self.train_ds = None
        self.train_idxs = None
        self.y_train_ds = None
        # ===== validation set ====
        self.val_ds = None
        self.val_idxs = None
        self.y_val_ds = None
        # ======= test set ========
        self.test_ds = None
        self.test_idxs = None
        self.y_test_ds = None

    def build(self):
        self.check_hw_accel()

        self.load_dataset()
        
    def load_json(self, json_file_path):
        with open(json_file_path, 'r') as file:
            return json.load(file)

    def check_hw_accel(self, v=True):
        gpu = len(tf.config.list_physical_devices('GPU'))>0
        if gpu: 
            # Set the device to GPU 
            cpu = tf.config.list_physical_devices('CPU')[0]
            gpu = tf.config.list_physical_devices('GPU')[0]
            tf.config.set_visible_devices([gpu,cpu])
            tf.config.experimental.set_memory_growth(gpu, True)
            if v:
                print("[experiment] GPU acceleration available.") 
    
    def load_dataset(self):
        if self.dataset is None:
            self.dataset = RichHDF5Dataset(self.dataset_h5, self.pkl_framesmap)
        
        return self.dataset

    def split_dataset(self, aug_train_ds=True):
        # Gathering the needed settings and data
        split_ratios = self.settings['ds_split_ratios']
        ds_reduction = self.settings['ds_reduction']
        nn_batch_size = self.settings['nn_batch_size']

        # Splitting the dataset into train, (validation) and test sets
        self.train_idxs, self.val_idxs, self.test_idxs, self.ds_infos = split_strategy(self.dataset, 
                                                                        ratios=split_ratios, 
                                                                        pkl_file=self.pkl_centersdict, 
                                                                        rseed=self.seed)
        
        # Reduce the dataset size if requested
        if ds_reduction > 0:
            self.train_idxs, self.val_idxs, self.test_idxs = reduce_sets(self.train_idxs, 
                                                                self.val_idxs, 
                                                                self.test_idxs, 
                                                                ds_reduction)
        
        # Create the train, (val) and test sets to feed the neural networks
        self.train_ds = HDF5Dataset(self.dataset, self.train_idxs, nn_batch_size, augmentation=aug_train_ds)
        self.val_ds = HDF5Dataset(self.dataset, self.val_idxs, nn_batch_size)
        self.test_ds = HDF5Dataset(self.dataset, self.test_idxs, nn_batch_size)
    
    def compute_class_weight(self, v=True):
        if self.ds_infos is not None:
            # Retrieves the dataset's labels
            ds_labels = self.ds_infos['labels']

            # Extract the train, val and test set labels
            self.y_train_ds = np.array(ds_labels)[self.train_idxs]
            self.y_val_ds = np.array(ds_labels)[self.val_idxs]
            self.y_test_ds = np.array(ds_labels)[self.test_idxs]

            # Calculate class balance using 'compute_class_weight'
            train_class_weights = compute_class_weight('balanced', 
                                                 classes=np.unique(self.y_train_ds), 
                                                 y=self.y_train_ds)

            # Create a dictionary that maps classes to their weights
            train_class_weight_dict = dict(enumerate(train_class_weights))

            if v:
                print("[experiment] class weights: ", train_class_weight_dict)

            self.train_class_weights = train_class_weight_dict
        else:
            raise Exception('[experiment] dataset not yet splitted.')

    def plot_split_charts(self, charts=[0, 0, 0]):
        if self.ds_infos is not None:
            if charts[0]:
                print_split_diagnostic_info(self.ds_infos)

            if charts[1]:
                plot_fsplit_info(self.ds_infos, log_scale=True)

            if charts[2]:
                plot_psplit_info(self.ds_infos)
        else:
            raise Exception('[experiment] dataset not yet splitted.')

    def nn_model(self):
        net_type = self.settings['nn_type']

        if net_type == 'obd':
            dropout = self.settings['nn_dropout']
            hidden_size = self.settings['obd_hidden_size']
            
            net_object = Net(ds_img_size = self.ds_img_size, 
                             ds_num_ch = self.ds_num_channels, 
                             ds_num_classes = self.ds_num_classes,
                             nn_dropout = dropout,
                             hidden_size = hidden_size)
            
            model = net_object.build(net_type)
        else:
            dropout = self.settings['nn_dropout']
            nn_activation = self.settings['nn_activation']
            nn_final_activation = self.settings['nn_final_activation']
            clm_use_tau = self.settings['clm_use_tau']

            net_object = Net(ds_img_size = self.ds_img_size, 
                             ds_num_ch = self.ds_num_channels, 
                             ds_num_classes = self.ds_num_classes,
                             nn_dropout = dropout,
                             nn_activation = nn_activation,
                             nn_final_activation = nn_final_activation,
                             clm_use_tau = clm_use_tau)

            model = net_object.build(net_type)

        return model
    
    def nn_model_compile(self, model, summary=False):
        '''
        "loss": "ODL",
        "metrics": ["ccr", "accuracy_oneoff", "mae", "ms"],
        "optimizer": "Adam",
        "learning_rate": 1e-3
        '''
        loss = self.settings['loss']
        metrics = self.settings['metrics']
        optimizer = self.settings['optimizer']
        learning_rate = self.settings['learning_rate']
        
        # ======= optimizer =======
        if optimizer == 'Adam':
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
        
        # ========= loss ==========
        if loss == 'ODL':
            loss = ordinal_distance_loss(self.ds_num_classes)
        elif loss == 'CCE':
            loss = tf.keras.losses.CategoricalCrossentropy()
        elif loss == 'QWK':
            cost_matrix = K.constant(make_cost_matrix(self.ds_num_classes), dtype=K.floatx())
            loss = qwk_loss(cost_matrix)
        
        # ======== metrics ========
        my_metrics = MyMetrics()
        metrics = [getattr(my_metrics, metric_name) for metric_name in metrics]

        # ======== compile ========
        model.compile(optimizer=optimizer, 
                      loss=loss, 
                      metrics=metrics)

        # Print model summary
        if summary:
            model.summary()

    def nn_model_train(self, model):
        epochs = self.settings['nn_epochs']

        history = model.fit(self.train_ds, 
                            shuffle=True,
                            epochs=epochs,
                            class_weight=self.train_class_weights,
                            validation_data=self.val_ds
                            )
        
        return history