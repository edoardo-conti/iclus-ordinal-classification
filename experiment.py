import os
import shutil
import json
import csv
from datetime import datetime
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from keras import backend as K
from keras import mixed_precision
from keras.callbacks import TensorBoard, BackupAndRestore, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from dataset import HDF5Dataset, TFDataset, load_ds_metadata, split_by_patients, split_by_centers, rus, trim_sets
from utilities import save_split_data, load_split_data, plot_charts
from network import NeuralNetwork
from losses import make_cost_matrix, qwk_loss, ordinal_distance_loss
from metrics import Metrics
from callbacks import GradCAMCallback, UpdataRichStatusBarCallback

class Experiment:
    def __init__(self, 
                 exps_json_path:str,
                 dataset_path:str,
                 ds_map_pkl:str,
                 ds_split_pkl:str,
                 results_dir:str,
                 eval_only:bool = False,
                 workers:int = 1,
                 shuffle_buffer_size:int = 100,
                 max_queue_size:int = 512,
                 verbose:int = 1,
                 output_mode:Tuple[int, int] = (0, 1),
                 input_size:int = 224,
                 num_channels:int = 3,
                 num_classes:int = 4,
                 random_state:int = 42):
        self.exps_json_path = exps_json_path
        self.dataset_h5 = dataset_path
        self.ds_map_pkl = ds_map_pkl
        self.ds_split_pkl = ds_split_pkl
        self.results_dir = results_dir
        self.eval_only = eval_only
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_queue_size = max_queue_size
        self.workers = workers
        self.verbose = verbose
        self.output_mode = output_mode
        self.seed = random_state

        # params to be computed
        self.device = None
        self.settings = None
        self.exp_name = ''
        self.exp_results_subdir = ''
        self.csv_results_path = ''
        self.csv_columns = []

        # =========================
        # ======== dataset ========
        # =========================
        self.ds_img_size = input_size
        self.ds_img_channels = num_channels
        self.ds_num_classes = num_classes
        self.dataset = None
        self.dataset_labels = None
        self.dataset_metadata = None
        self.train_class_weights = None
        # ======= train set =======
        self.idxs_train = None
        self.x_train = None
        self.y_train = None
        # ===== validation set ====
        self.idxs_val = None
        self.x_val = None
        self.y_val = None
        # ======= test set ========
        self.idxs_test = None
        self.x_test = None
        self.y_test = None
        
        # =========================
        # ==== neural network =====
        # =========================
        self.last_conv_layer = None
        self.train_metrics_exl = ['top_2_acc', 'top_3_acc', 'ms', 'spearman', 'qwk']
        self.metrics_results = {}
    

    def build(self):
        # setting os environment
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        
        #TODO: da attivare
        #tf.config.optimizer.set_jit(True)
        #policy = mixed_precision.Policy("mixed_float16")
        #mixed_precision.set_global_policy(policy)

        self.is_gpu_available()
        self.load_dataset()
        self.init_results()
    

    def load_json(self, json_file_path):
        with open(json_file_path, 'rb') as file:
            return json.load(file)


    def is_gpu_available(self):
        cpu_device = tf.config.list_physical_devices('CPU')[0]
        gpu_devices = tf.config.list_physical_devices('GPU')
        
        if gpu_devices:
            # "/GPU:0"
            self.device = gpu_devices[0]
        else:
            self.device = cpu_device


    def load_exp_settings(self, exp_idx):
        # load the requested experiment settings by using the index to find it in the json file
        self.settings = self.load_json(self.exps_json_path)[exp_idx]
        
        # build the experiment name based on the configuration extracted
        self.exp_name = self.build_exp_name()
        
        # create the experiment results subdirectory
        self.exp_results_subdir = os.path.join(self.results_dir, self.exp_name)
        os.makedirs(self.exp_results_subdir, exist_ok=True)

        # create the weights subdirectory
        weights_path = os.path.join(self.exp_results_subdir, 'weights/') 
        os.makedirs(weights_path, exist_ok=True)

        # reset previous configurations
        self.metrics_results = {}
        self.metrics_results['experiment'] = self.exp_name
        
        return self.exp_name


    def build_exp_name(self):
        # parameters not to be used to generate the experiment name
        # excl_params = ["ds_split_ratio", "ds_trim", "metrics"]
        excl_params = ["ds_split_ratio", "metrics"]
        experiment_params = {key: value for key, value in self.settings.items() if key not in excl_params}

        # generate the experiment name based on the parameters
        exp_name = "_".join(str(value) for value in experiment_params.values())

        return exp_name
    

    def load_dataset(self):
        self.dataset = HDF5Dataset(self.dataset_h5, self.ds_map_pkl)
        self.dataset_labels = load_ds_metadata(self.dataset, self.ds_split_pkl, only_labels=True)


    def init_results(self):
        # declare the CSV file path and columns
        self.csv_results_path = os.path.join(self.results_dir, "results.csv")
        self.csv_columns = ["experiment", "ccr", "mae", "ms", "rmse", "acc_1off", "qwk"]
        
        # ask for a confirmation
        confirmation = input("Do you want to delete existing run files? (y/[N]): ").lower()
        
        # experiments checkpoint file 
        if not os.path.exists('run_checkpoint.txt') and confirmation == 'y':
            # clean the previous results and re-make the directory
            if os.path.exists(self.results_dir):
                shutil.rmtree(self.results_dir)
            os.makedirs(self.results_dir)
            
            # remove previous logs/ directory
            if os.path.exists('./logs/'):
                shutil.rmtree('./logs/')

            # delete splitting file
            if os.path.exists("./splitdata.pkl"):
                os.remove("./splitdata.pkl")

            # create the CSV file and write the header
            with open(self.csv_results_path, mode='w', encoding='UTF-8', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(self.csv_columns)
        
        if self.eval_only:
            # delete splitting file
            if os.path.exists("./splitdata.pkl"):
                os.remove("./splitdata.pkl")

    
    def check_exps_ckp(self, curr_exp):
        ret_val = False

        if os.path.exists('run_checkpoint.txt'):
            with open('run_checkpoint.txt', 'r', encoding='UTF-8') as ckp_file:
                for line in ckp_file:
                    ckp_exp = line.split()[0]
                    if ckp_exp == curr_exp:
                        ret_val = True

        return ret_val


    def split_dataset(self, exps_common_settings=None):
        # gather the needed settings
        settings_params = ['ds_split_ratio', 'ds_trim', 'ds_split_by', 'ds_us']
        settings = exps_common_settings if exps_common_settings else self.settings
        split_ratio, ds_trim, split_by, ds_us = [settings[key] for key in settings_params]
        
        # split the dataset into train, (validation) and test sets
        split_method = split_by_patients if split_by == 'patient' else split_by_centers
        split = split_method(self.dataset, ratio=split_ratio, pkl_file=self.ds_split_pkl, rseed=self.seed)
        self.idxs_train, self.idxs_val, self.idxs_test, self.dataset_metadata = split
        
        # training set random undersampling
        if ds_us:
            y_train = self.dataset_labels[self.idxs_train]
            if 'rus' in ds_us:
                self.idxs_train, self.y_train = rus(self.idxs_train, y_train, strategy=ds_us, rseed=self.seed)
        
        # trim the dataset if requested
        if ds_trim > 0:
            trimmed_sets = trim_sets(self.idxs_train, self.idxs_test, self.idxs_val, ds_trim, self.seed)
            self.idxs_train, self.idxs_val, self.idxs_test = trimmed_sets
        
        # extract the train, val and test set labels
        self.y_train = self.dataset_labels[self.idxs_train]
        self.y_val = self.dataset_labels[self.idxs_val]
        self.y_test = self.dataset_labels[self.idxs_test]
        
        # export the splitting
        split_data = {'train': {'x': self.idxs_train, 'y': self.y_train}, 
                        'val': {'x': self.idxs_val, 'y': self.y_val}, 
                        'test': {'x': self.idxs_test, 'y': self.y_test}, 
                        'metadata': self.dataset_metadata}
        save_path = self.results_dir if exps_common_settings else self.exp_results_subdir
        save_split_data(split_data, save_path)

    
    def load_dataset_splitted(self):
        ltrain, lval, ltest, lmetadata = load_split_data(self.results_dir)
        self.idxs_train, self.y_train = ltrain['x'], ltrain['y']
        self.idxs_val, self.y_val = lval['x'], lval['y']
        self.idxs_test, self.y_test = ltest['x'], ltest['y']
        self.dataset_metadata = lmetadata


    def generate_sets(self):
        # gather the needed settings and data
        batch_size = self.settings['nn_batch_size']
        augmentation = self.settings['augmentation']
        
        # create the train, (val) and test sets to feed the neural networks
        self.x_train = TFDataset(self.dataset, 
                                 self.idxs_train, 
                                 batch_size=batch_size, 
                                 buffer_size=self.shuffle_buffer_size, 
                                 is_train=True,
                                 augmentation=augmentation,
                                 device=self.device).as_iterator()
        self.x_val = TFDataset(self.dataset, self.idxs_val, batch_size=batch_size, device=self.device).as_iterator()
        self.x_test = TFDataset(self.dataset, self.idxs_test, batch_size=batch_size, device=self.device).as_iterator()
        
        
    def compute_class_weight(self):
        # calculate class balance using 'compute_class_weight'
        train_class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        train_class_weights = np.round(train_class_weights, 4)
        self.train_class_weights = dict(enumerate(train_class_weights))
        return self.train_class_weights
    
    def generate_split_charts(self, charts=None, per_exp=False):
        if self.dataset_metadata is not None:
            # default graphs
            charts = charts or ["pdistr", "lsdistr_pie", "ldistr"]
            
            # get the output mode
            display, save = self.output_mode

            # select the path to save the graphs (global or per experiment)
            save_path = self.exp_results_subdir if per_exp else self.results_dir
            
            plot_charts(self, charts, display, save, save_path)
        else:
            raise Exception('dataset not yet splitted.')


    def nn_model_build(self):
        # get the network type: obd, clm, resnet18, cnn128, vgg16
        net_type = self.settings['nn_type']
        
        # get the common networks parameters
        common_params = {
            'ds_img_size': self.ds_img_size,
            'ds_img_channels': self.ds_img_channels,
            'ds_num_classes': self.ds_num_classes,
            'nn_dropout': self.settings['nn_dropout']
        }

        if net_type == 'obd':
            net_object = NeuralNetwork(nn_backbone = self.settings['nn_backbone'],
                                       obd_hidden_size=self.settings['obd_hidden_size'], 
                                       **common_params)
        elif net_type == 'clm':
            net_object = NeuralNetwork(nn_backbone = self.settings['nn_backbone'],
                                       clm_link=self.settings['clm_link'], 
                                       clm_use_tau=self.settings['clm_use_tau'], 
                                       **common_params)
        else:
            net_object = NeuralNetwork(**common_params)

        # building the defined neural network model 
        model = net_object.build(net_type)

        # auto-search the last convolutional layer of the model (uselful for GRAD-cams)
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                self.last_conv_layer = layer.name
                break

        return model
    

    def nn_model_compile(self, model, summary=False):
        loss = self.settings['loss']
        metrics = self.settings['metrics']
        optimizer = self.settings['optimizer']
        learning_rate = self.settings['learning_rate']
        
        # optimizer
        if optimizer == 'Adam':
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate,
                                                       decay=self.settings['weight_decay'],
                                                       momentum=self.settings['momentum'])
        
        # loss function
        if loss == 'ODL':
            loss = ordinal_distance_loss(self.ds_num_classes)
        elif loss == 'CCE':
            loss = tf.keras.losses.CategoricalCrossentropy()
        elif loss == 'QWK':
            cost_matrix = K.constant(make_cost_matrix(self.ds_num_classes), dtype=K.floatx())
            loss = qwk_loss(cost_matrix)
        
        # metrics
        metrics_t = Metrics(self.ds_num_classes, self.settings['nn_type'])
        train_metrics = [getattr(metrics_t, metric_name) for metric_name in metrics if metric_name not in self.train_metrics_exl]

        # compile
        model.compile(optimizer=optimizer, loss=loss, metrics=train_metrics)

        # print model summary
        if summary:
            model.summary()


    def nn_model_train(self, model, gradcam_freq=5, status_bar=None):
        # parameters
        epochs = self.settings['nn_epochs']
        batch_size = self.settings['nn_batch_size']
        ckpt_filename = os.path.join(self.exp_results_subdir, "weights", "best_ckpt.hdf5")

        # callbacks
        tensorboard = TensorBoard(log_dir=f"logs/fit/{self.exp_name}", histogram_freq=1)
        backup = BackupAndRestore(backup_dir="backup/")
        checkpoint = ModelCheckpoint(ckpt_filename, monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=self.verbose)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=self.verbose)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=6, min_lr=1e-6, verbose=self.verbose)
        gradcam = GradCAMCallback(model, self, freq=gradcam_freq) if gradcam_freq > 0 else None
        update_status = UpdataRichStatusBarCallback(status_bar, epochs) if status_bar is not None else None

        # build callbacks list
        callbacks = [tensorboard, backup, checkpoint, early_stop, reduce_lr, gradcam, update_status]
        callbacks = [callback for callback in callbacks if callback is not None]
        
        # neural network fit
        history = model.fit(self.x_train,
                            epochs=epochs,
                            steps_per_epoch=len(self.idxs_train) // batch_size,
                            class_weight=self.train_class_weights,
                            validation_data=self.x_val,
                            validation_steps=len(self.idxs_val) // batch_size,
                            callbacks=callbacks,
                            verbose=self.verbose,
                            max_queue_size=self.max_queue_size,
                            workers=self.workers,
                            use_multiprocessing=False
                            )
        
        return history
    

    def nn_train_graphs(self, history):
        # get the output mode
        display, save = self.output_mode

        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # plot loss functions
        for loss in history.history.keys():
            if loss.endswith('loss'):
                label = loss
                linestyle = '--' if loss.startswith('val_') else '-'
                ax1.plot(history.history[loss], label=label, linestyle=linestyle)
        
        ax1.legend()
        ax1.set_xlabel('epoch')
        ax1.set_ylabel(f'{self.settings["loss"]}')
        ax1.set_title(f'Loss - {self.exp_name}')
        ax1.grid()

        # plot metrics
        for metric in history.history.keys():
            if not metric.endswith('loss'):
                label = metric
                linestyle = '--' if metric.startswith('val_') else '-'
                ax2.plot(history.history[metric], label=label, linestyle=linestyle)

        ax2.legend()
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('metric')
        ax2.set_title(f'Metrics - {self.exp_name}')
        ax2.grid()

        # save the training charts
        if save:
            train_graphs_path = os.path.join(self.exp_results_subdir, "training_plot.png")
            plt.savefig(train_graphs_path)
        
        # Show the figure
        if display:
            plt.show()
        
        plt.close()


    def nn_model_evaluate(self, model, weights=None, load_best_weights=True):
        # if specified, load the weights passed as argument (priority)
        if weights is not None:
            model.load_weights(weights)
        elif load_best_weights:
            # load the best weights
            best_weights_file = os.path.join(self.exp_results_subdir, "weights", "best_ckpt.hdf5")
            try:
                model.load_weights(best_weights_file)
            except Exception as e:
                raise Exception('error while loading best weights file: ', e)
        
        # get the batch size
        nn_batch_size = self.settings['nn_batch_size']
        
        # model evaluation, get the predictions by running the model inference
        y_test_pred = model.predict(self.x_test, 
                                    steps=-(-len(self.idxs_test) // nn_batch_size),
                                    verbose=self.verbose,
                                    max_queue_size=self.max_queue_size,
                                    workers=self.workers,
                                    use_multiprocessing=False
                                    )
        
        # compute evaluation metrics
        metrics_e = Metrics(self.ds_num_classes, self.settings['nn_type'])
        eval_metrics = [(getattr(metrics_e, metric_name), metric_name) for metric_name in self.settings['metrics']]
        
        for metric, metric_name  in eval_metrics:
            result = metric(self.y_test, y_test_pred)
            result = np.round(result, 4)
            self.metrics_results[metric_name] = result
        
        # test set confusion matrix
        display, save = self.output_mode
        cfmat_fig = metrics_e.confusion_matrix(self.y_test, y_test_pred, show=display)
        if save:
            cfmat_fig_path = os.path.join(self.exp_results_subdir, "confusion_matrix.png")
            cfmat_fig.savefig(cfmat_fig_path)
        plt.close()
        
        # save results on the csv        
        values_columns = [self.metrics_results.get(column, '-') for column in self.csv_columns]
        with open(self.csv_results_path, mode='a', encoding='UTF-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(values_columns)
        
        # update the experiments checkpoint file 
        datetime_log = datetime.now().strftime("%m/%d/%Y,%H:%M:%S")
        with open('run_checkpoint.txt', mode='a', encoding='UTF-8') as ckp_file:
            ckp_file.write(f"{self.exp_name} {datetime_log}\n")