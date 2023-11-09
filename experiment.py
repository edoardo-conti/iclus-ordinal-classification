import os
import shutil
import json
import csv
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from dataset import RichHDF5Dataset, split_strategy, trim_sets, create_tf_dataset
from utilities import print_split_ds_info, plot_frames_split, plot_patients_split, plot_labels_distr
from network import NeuralNetwork
from losses import make_cost_matrix, qwk_loss, ordinal_distance_loss
from metrics import Metrics
from gradcam import GradCAMCallback

class Experiment:
    def __init__(self, 
                 exps_json_path:str,
                 dataset_path:str,
                 ds_map_pkl:str,
                 ds_split_pkl:str,
                 results_dir:str,
                 workers:int = 1,
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
        self.max_queue_size = max_queue_size
        self.workers = workers
        self.verbose = verbose
        self.output_mode = output_mode
        self.seed = random_state

        # params to be computed
        self.hw_accel = False
        self.settings = None
        self.exp_name = ''
        self.exp_results_subdir = None
        self.csv_results_path = None
        self.csv_columns = []

        # =========================
        # ======== dataset ========
        # =========================
        self.ds_img_size = input_size
        self.ds_num_channels = num_channels
        self.ds_num_classes = num_classes
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
        
        # =========================
        # ==== neural network =====
        # =========================
        self.last_conv_layer = None
        self.train_metrics_exl = ['top_2_acc', 'top_3_acc', 'qwk', 'spearman']
        self.metrics_results = {}
    
    def build(self):
        # setting os environment
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        self.check_hw_accel()
        self.load_dataset()
        self.init_results()
    
    def load_json(self, json_file_path):
        with open(json_file_path, 'rb') as file:
            return json.load(file)

    def check_hw_accel(self):
        gpu_count = len(tf.config.list_physical_devices('GPU')) > 0
        if gpu_count: 
            # Set the device to GPU 0
            cpu = tf.config.list_physical_devices('CPU')[0]
            gpu = tf.config.list_physical_devices('GPU')[0]
            tf.config.set_visible_devices([gpu,cpu])
            tf.config.experimental.set_memory_growth(gpu, True)

            # set the class attribute regarding the hw acceleration 
            self.hw_accel = True
       
    def load_exp_settings(self, exp_idx):
        # load the requested experiment settings by using the index to find it in the json file
        self.settings = self.load_json(self.exps_json_path)[exp_idx]
        
        # build the experiment name based on the configuration extracted
        self.exp_name = self.build_exp_name()

        # create the experiment results subdirectory
        self.exp_results_subdir = os.path.join(self.results_dir, self.exp_name)
        if not os.path.exists(self.exp_results_subdir):
            os.makedirs(self.exp_results_subdir)

        return self.exp_name

    def build_exp_name(self):
        excl_params = ["ds_split_ratios", "ds_trim", "metrics"]
        experiment_params = {key: value for key, value in self.settings.items() if key not in excl_params}

        # Generate the experiment name based on the parameters
        exp_name = "_".join(str(value) for value in experiment_params.values())

        return exp_name
        
    def load_dataset(self):
        if self.dataset is None:
            self.dataset = RichHDF5Dataset(self.dataset_h5, self.ds_map_pkl)

    def init_results(self):
        # clean the previous results and re-make the directory
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir)

        # declare the CSV file path
        self.csv_results_path = os.path.join(self.results_dir, "results.csv")
        # setup the columns for the CSV
        self.csv_columns = ["experiment", "ccr", "mae", "ms", "rmse", "acc_1off", "spearman", "qwk"]
        # create the CSV file and write the header
        with open(self.csv_results_path, mode='w', encoding='UTF-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(self.csv_columns)

    def split_dataset(self, exps_common_settings=None):
        # Gathering the needed settings
        if exps_common_settings is None:
            split_ratios = self.settings['ds_split_ratios']
            ds_trim = self.settings['ds_trim']
        else:
            split_ratios = exps_common_settings['ds_split_ratios']
            ds_trim = exps_common_settings['ds_trim']

        # Splitting the dataset into train, (validation) and test sets
        self.train_idxs, self.val_idxs, self.test_idxs, self.ds_infos = split_strategy(self.dataset, 
                                                                                        ratios=split_ratios, 
                                                                                        pkl_file=self.ds_split_pkl, 
                                                                                        rseed=self.seed)
        
        # Trim the dataset size if requested
        if ds_trim < 1.0:
            self.train_idxs, self.val_idxs, self.test_idxs = trim_sets(self.train_idxs,
                                                                            self.val_idxs,
                                                                            self.test_idxs,
                                                                            ds_trim)

    def generate_sets(self, aug_train_ds=True):
        # Gathering the needed settings and data
        nn_batch_size = self.settings['nn_batch_size']

        # Create the train, (val) and test sets to feed the neural networks
        #self.train_ds = HDF5Dataset(self.dataset, self.train_idxs, batch_size=nn_batch_size, augmentation=aug_train_ds).create_dataset()
        #self.val_ds = HDF5Dataset(self.dataset, self.val_idxs, batch_size=nn_batch_size).create_dataset()
        #self.test_ds = HDF5Dataset(self.dataset, self.test_idxs, batch_size=nn_batch_size).create_dataset()

        self.train_ds = create_tf_dataset(self.dataset, self.train_idxs, batch_size=nn_batch_size, is_training=True, aug=aug_train_ds)
        self.val_ds = create_tf_dataset(self.dataset, self.val_idxs, batch_size=nn_batch_size)
        self.test_ds = create_tf_dataset(self.dataset, self.test_idxs, batch_size=nn_batch_size)
        
    def compute_class_weight(self):
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

            self.train_class_weights = dict(enumerate(train_class_weights))
        else:
            raise Exception('dataset not yet splitted.')
    
    def generate_split_charts(self, charts=None, per_exp=False):
        if self.ds_infos is not None:
            if charts is None:
                charts = ["fdistr", "pdistr", "ldistr"]
            
            # get the output mode
            display, save = self.output_mode

            # choose the right save path (global of per-experiment)
            save_path = self.exp_results_subdir if per_exp else self.results_dir
                
            if "splitinfo" in charts:
                print_split_ds_info(self.ds_infos)
            
            if "fdistr" in charts:
                pfs = plot_frames_split(self.ds_infos, log_scale=True, display=display)
                if save:
                    chart_file_path = os.path.join(save_path, "split_per_frames.png")
                    pfs.savefig(chart_file_path)
                    plt.close()

            if "pdistr" in charts:
                pps = plot_patients_split(self.ds_infos, display=display)
                if save:
                    chart_file_path = os.path.join(save_path, "split_per_patients.png")
                    pps.savefig(chart_file_path)
                    plt.close()

            if "ldistr" in charts:
                pld = plot_labels_distr(self.y_train_ds, self.y_val_ds, self.y_test_ds, display=display)
                if save:
                    chart_file_path = os.path.join(save_path, "distr_labels_per_set.png")
                    pld.savefig(chart_file_path)
                    plt.close()
        else:
            raise Exception('dataset not yet splitted.')

    def nn_model_build(self):
        net_type = self.settings['nn_type']
        dropout = self.settings['nn_dropout']

        if net_type == 'obd':
            net_backbone = self.settings['nn_backbone']
            hidden_size = self.settings['obd_hidden_size']
            
            net_object = NeuralNetwork(ds_img_size = self.ds_img_size,
                             ds_num_ch = self.ds_num_channels,
                             ds_num_classes = self.ds_num_classes,
                             nn_backbone = net_backbone,
                             nn_dropout = dropout,
                             obd_hidden_size = hidden_size)

        elif net_type == 'clm':
            net_backbone = self.settings['nn_backbone']
            clm_link_function = self.settings['clm_link_function']
            clm_use_tau = self.settings['clm_use_tau']

            net_object = NeuralNetwork(ds_img_size = self.ds_img_size,
                             ds_num_ch = self.ds_num_channels,
                             ds_num_classes = self.ds_num_classes,
                             nn_backbone = net_backbone,
                             nn_dropout = dropout,
                             clm_link_function = clm_link_function,
                             clm_use_tau = clm_use_tau)

        else:
            net_object = NeuralNetwork(ds_img_size = self.ds_img_size,
                             ds_num_ch = self.ds_num_channels,
                             ds_num_classes = self.ds_num_classes,
                             nn_dropout = dropout)

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
        weight_decay = self.settings['weight_decay']
        
        # ============== optimizer ==============
        if optimizer == 'Adam':
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, 
                                                        weight_decay=weight_decay)
        else:
            optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, 
                                                       momentum=self.settings['momentum'])
        
        # ================ loss =================
        if loss == 'ODL':
            loss = ordinal_distance_loss(self.ds_num_classes)
        elif loss == 'CCE':
            loss = tf.keras.losses.CategoricalCrossentropy()
        elif loss == 'QWK':
            cost_matrix = K.constant(make_cost_matrix(self.ds_num_classes), dtype=K.floatx())
            loss = qwk_loss(cost_matrix)
        
        # =============== metrics ===============
        metrics_t = Metrics(self.ds_num_classes, self.settings['nn_type'], 'train')
        #train_metrics = [getattr(metrics_t, metric_name) for metric_name in metrics if metric_name not in train_metrics_exl]

        train_metrics = []
        for metric_name in metrics:
            if metric_name not in self.train_metrics_exl:
                try:
                    metric = getattr(metrics_t, metric_name)
                except AttributeError:
                    metric = metric_name
                train_metrics.append(metric)

        # =============== compile ===============
        model.compile(optimizer=optimizer, loss=loss, metrics=train_metrics)

        # Print model summary
        if summary:
            model.summary()

    def nn_model_train(self, model, gradcam_freq=5, gradcam_show=False):
        # parameters
        epochs = self.settings['nn_epochs']
        batch_size = self.settings['nn_batch_size']

        # =============== Callbacks ===============
        # ModelCheckpoint, saving the best model
        checkpoint = ModelCheckpoint(f'weights/{self.exp_name}', monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=self.verbose)

        # EarlyStopping, stop training when model has stopped improving
        early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=self.verbose)
        
        # ReduceLROnPlateau, reduce learning rate when model has stopped improving.
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=self.verbose)

        # # GRAD-Cam, showing the gradients activation maps
        # gradcams_dir_path = os.path.join(self.exp_results_subdir, "gradcams/")
        # if not os.path.exists(gradcams_dir_path):
        #     os.makedirs(gradcams_dir_path)
        # gradcam = GradCAMCallback(model, self.last_conv_layer, self.val_ds, freq=gradcam_freq, show_cams=gradcam_show, save_cams=gradcams_dir_path)
        
        # # callbacks list
        # callbacks = [checkpoint, early_stop, reduce_lr] + [gradcam] * (gradcam_freq > 0)
        callbacks = [checkpoint, early_stop, reduce_lr]
        
        # steps
        train_samples = len(self.train_idxs)
        train_steps = train_samples // batch_size
        if train_samples % batch_size != 0:
            train_steps += 1
        
        val_samples = len(self.val_idxs)
        val_steps = val_samples // batch_size
        if val_samples % batch_size != 0:
            val_steps += 1

        # =============== Neural Network Fit ===============
        history = model.fit(self.train_ds.as_numpy_iterator(), 
                            shuffle=True,
                            epochs=epochs,
                            steps_per_epoch=train_steps,
                            class_weight=self.train_class_weights,
                            validation_data=self.val_ds.as_numpy_iterator(),
                            validation_steps=val_steps,
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

        # Create a figure with two subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Get the loss and metrics from history
        history_keys = history.history.keys()

        # Loss subplot
        for loss in history_keys:
            if loss.endswith('loss'):
                label = loss
                if loss.startswith('val_'):
                    ax1.plot(history.history[loss], label=label, linestyle='--')
                else:
                    ax1.plot(history.history[loss], label=label)
        ax1.legend()
        ax1.set_xlabel('epoch')
        ax1.set_ylabel(f'{self.settings["loss"]}')
        ax1.set_title('Loss - ' + self.exp_name)
        ax1.grid()

        # Metrics subplot
        for metric in history_keys:
            if not metric.endswith('loss'):
                label = metric
                if metric.startswith('val_'):
                    ax2.plot(history.history[metric], label=label, linestyle='--')
                else:
                    ax2.plot(history.history[metric], label=label)
        ax2.legend()
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('metric')
        ax2.set_title('Metrics - ' + self.exp_name)
        ax2.grid()

        # Show the figure
        if display:
            plt.show()

        if save:
            train_graphs_path = os.path.join(self.exp_results_subdir, "training_plot.png")
            plt.savefig(train_graphs_path)
        
        plt.close()

    def nn_model_evaluate(self, model, load_best_weights=True):
        # Load the best weights
        if load_best_weights:
            model.load_weights(f'weights/{self.exp_name}')

        # steps
        nn_batch_size = self.settings['nn_batch_size']
        test_samples = len(self.test_idxs)
        test_steps = test_samples // nn_batch_size
        if test_samples % nn_batch_size != 0:
            test_steps += 1

        # TODO: check if evaluate give same results as manual metrics computing
        model.evaluate(self.test_ds.as_numpy_iterator(), 
                       steps=test_steps,
                       max_queue_size=self.max_queue_size,
                       workers=self.workers,
                       use_multiprocessing=False,
                       verbose=self.verbose)
        
        # get the predictions by running the model inference
        y_test_pred = model.predict(self.test_ds.as_numpy_iterator(), 
                                    steps=test_steps,
                                    max_queue_size=self.max_queue_size,
                                    workers=self.workers,
                                    use_multiprocessing=False,
                                    verbose=self.verbose)

        # compute the evaluation metrics
        metrics_e = Metrics(self.ds_num_classes, self.settings['nn_type'], 'eval')
        #eval_metrics = [getattr(metrics_e, metric_name) for metric_name in ]
        
        eval_metrics = []
        for metric_name in self.settings['metrics']:
            try:
                metric = getattr(metrics_e, metric_name)
                metric_name = metric.__name__
                is_function = True
            except AttributeError:
                metric = metric_name
                is_function = False

            eval_metrics.append((metric, metric_name, is_function))
        
        # compute all the metrics
        for metric, metric_name, is_function in eval_metrics:
            if is_function:
                result = metric(self.y_test_ds, y_test_pred)
                result = np.round(result, 4)
                self.metrics_results[metric_name] = result
            else:
                # TODO: string metric, idk how to do it
                pass
        
        # Test Set Confusion Matrix
        # get the output mode
        display, save = self.output_mode
        cfmat_fig = metrics_e.confusion_matrix(self.y_test_ds, y_test_pred, show=display)
        if save:
            cfmat_fig_path = os.path.join(self.exp_results_subdir, "confusion_matrix.png")
            cfmat_fig.savefig(cfmat_fig_path)
        plt.close()

        # Save results on the csv
        # add the experiment name to the result dictionary
        self.metrics_results['experiment'] = self.exp_name
        # get the list of values to insert into the columns
        values_columns = [self.metrics_results.get(column, '-') for column in self.csv_columns]
        
        # add data to CSV file
        with open(self.csv_results_path, mode='a', encoding='UTF-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(values_columns)