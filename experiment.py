import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from dataset import RichHDF5Dataset, HDF5Dataset, split_strategy, reduce_sets
from utilities import print_split_ds_info, plot_fsplit_info, plot_psplit_info, plot_labels_distr
from network import NeuralNetwork
from losses import make_cost_matrix, qwk_loss, ordinal_distance_loss
from metrics import Metrics
from gradcam import GradCAMCallback

class Experiment:
    def __init__(self, 
                 exps_json_path:str,
                 exp_idx:int, 
                 dataset_path:str,
                 ds_map_pkl:str,
                 ds_split_pkl:str,
                 csv_results:str,
                 input_size:int = 224, 
                 num_channels:int = 3, 
                 num_classes:int = 4,
                 random_state:int = 42):
        self.dataset_h5 = dataset_path
        self.ds_map_pkl = ds_map_pkl
        self.ds_split_pkl = ds_split_pkl
        self.csv_results = csv_results
        self.seed = random_state
        self.settings = self.load_json(exps_json_path)[exp_idx]
        
        # params to be computed by building the class
        self.exp_name = ''
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
        self.train_metrics_exl = ['top_2_acc', 'top_3_acc', 'ms', 'qwk', 'spearman']
        self.metrics_results = {}

    def build(self):
        self.check_hw_accel()
        self.build_exp_name()
        self.load_dataset()
        self.load_results_csv()
        
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
    
    def build_exp_name(self, verbose=True):
        excl_params = ["ds_split_ratios", "ds_reduction", "metrics"]
        experiment_params = {key: value for key, value in self.settings.items() if key not in excl_params}

        # Generate the experiment name based on the parameters
        self.exp_name = "exp_" + "_".join(str(value) for value in experiment_params.values())

        if verbose:
            print(f"[experiment] experiment name built from parameters: {self.exp_name}") 

    def load_dataset(self):
        if self.dataset is None:
            self.dataset = RichHDF5Dataset(self.dataset_h5, self.ds_map_pkl)

    def load_results_csv(self):
        # setup the columns 
        self.csv_columns = ["experiment", "ccr", "acc_1off", "mae", "qwk"]
        # Check if the CSV file already exists
        if not os.path.exists(self.csv_results):
            # If the file does not exist, create it and write the header
            with open(self.csv_results, mode='w', encoding='UTF-8', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(self.csv_columns)

    def split_dataset(self, aug_train_ds=True):
        # Gathering the needed settings and data
        split_ratios = self.settings['ds_split_ratios']
        ds_reduction = self.settings['ds_reduction']
        nn_batch_size = self.settings['nn_batch_size']

        # Splitting the dataset into train, (validation) and test sets
        self.train_idxs, self.val_idxs, self.test_idxs, self.ds_infos = split_strategy(self.dataset, 
                                                                        ratios=split_ratios, 
                                                                        pkl_file=self.ds_split_pkl, 
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

    def plot_split_charts(self, splitinfo=True, fdistr=True, pdistr=True, ldistr=True):
        if self.ds_infos is not None:
            if splitinfo:
                print_split_ds_info(self.ds_infos)

            if fdistr:
                plot_fsplit_info(self.ds_infos, log_scale=True)
                
            if pdistr:
                plot_psplit_info(self.ds_infos)

            if ldistr:
                plot_labels_distr(self.y_train_ds, self.y_val_ds, self.y_test_ds)
            
        else:
            raise Exception('[experiment] dataset not yet splitted.')

    def nn_model_build(self, verbose=True):
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
            
            if verbose:
                print(f"[experiment] '{net_type}' neural network built with:")
                print(f"[experiment]\tbackbone -> {net_backbone}")
                print(f"[experiment]\tdropout -> {dropout}")
                print(f"[experiment]\tobd_hidden_size -> {hidden_size}\n")

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
            
            if verbose:
                print(f"[experiment] '{net_type}' neural network built with:")
                print(f"[experiment]\tbackbone -> {net_backbone}")
                print(f"[experiment]\tdropout -> {dropout}")
                print(f"[experiment]\tclm_link_function -> {clm_link_function}")
                print(f"[experiment]\tclm_use_tau -> {clm_use_tau}\n")

        else:
            net_object = NeuralNetwork(ds_img_size = self.ds_img_size, 
                             ds_num_ch = self.ds_num_channels, 
                             ds_num_classes = self.ds_num_classes,
                             nn_dropout = dropout)
            
            if verbose:
                print(f"[experiment] '{net_type}' neural network built with:")
                print(f"[experiment]\tdropout -> {dropout}\n")

        model = net_object.build(net_type)

        return model
    
    def nn_model_compile(self, model, summary=False, verbose=True):
        loss = self.settings['loss']
        metrics = self.settings['metrics']
        optimizer = self.settings['optimizer']
        learning_rate = self.settings['learning_rate']
        weight_decay = self.settings['weight_decay']
        
        # ======= optimizer =======
        if optimizer == 'Adam':
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, 
                                                        weight_decay=weight_decay)
        else:
            optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, 
                                                       momentum=self.settings['momentum'])
        
        # ========= loss ==========
        if loss == 'ODL':
            loss = ordinal_distance_loss(self.ds_num_classes)
        elif loss == 'CCE':
            loss = tf.keras.losses.CategoricalCrossentropy()
        elif loss == 'QWK':
            cost_matrix = K.constant(make_cost_matrix(self.ds_num_classes), dtype=K.floatx())
            loss = qwk_loss(cost_matrix)
        
        # ======== metrics ========
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

        # ======== compile ========
        model.compile(optimizer=optimizer, 
                      loss=loss, 
                      metrics=train_metrics)

        if verbose:
                print("[experiment] neural network model compiled with:")
                print(f"[experiment]\tloss -> {self.settings['loss']}")
                print(f"[experiment]\toptimizer -> {self.settings['optimizer']}")
                print(f"[experiment]\tlearning rate -> {self.settings['learning_rate']}\n")

        # Print model summary
        if summary:
            model.summary()

    def nn_model_train(self, model, gradcam_freq=5, verbose=True):
        # parameters
        epochs = self.settings['nn_epochs']

        # callbacks
        # saving the best model
        checkpoint = ModelCheckpoint(f'checkpoints/{self.exp_name}', save_weights_only=True, monitor='val_loss', verbose=1, save_best_only=True)
        
        # Grad-CAM
        # auto-search the last convolutional layer of the model
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break
        gradcam = GradCAMCallback(model, last_conv_layer, self.val_ds, freq=gradcam_freq)
        
        if verbose:
                print(f"[experiment] last model's convolutional layer extracted: {last_conv_layer} (will be used by Grad-CAM)\n")

        # neural network fit
        history = model.fit(self.train_ds, 
                            shuffle=True,
                            epochs=epochs,
                            class_weight=self.train_class_weights,
                            validation_data=self.val_ds,
                            callbacks=[checkpoint, gradcam]
                            )
        
        return history
    
    def nn_train_graphs(self, history):
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
        plt.show()

    def nn_model_evaluate(self, model, load_best_weights=True, cf_mat=True, save_on_csv=True, v=True):       
        # Load the best weights
        if load_best_weights:
            model.load_weights(f'checkpoints/{self.exp_name}')

            if v:
                print(f'[experiment] best model weights loaded.')

        # TODO: check if evaluate give same results as manual metrics computing
        model.evaluate(self.test_ds)

        # get the predictions by running the model inference
        y_test_pred = model.predict(self.test_ds)

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
                #result = tf.round(result, 3).numpy()
                result = np.round(result, 4)
                self.metrics_results[metric_name] = result
                print(f'{metric_name}: {result:.4f}')
            else:
                # TODO: string metric, idk how to do it
                pass
        
        # Test Set Confusion Matrix
        if cf_mat:
            metrics_e.confusion_matrix(self.y_test_ds, y_test_pred)

        # Save results on the csv if requested
        if save_on_csv:
            # add the experiment name to the result dictionary
            self.metrics_results['experiment'] = self.exp_name
            # get the list of values to insert into the columns
            values_columns = [self.metrics_results.get(column, '-') for column in self.csv_columns]

            # add data to CSV file
            with open(self.csv_results, mode='a', encoding='UTF-8', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(values_columns)

            print(f"[experiment] experiment results saved on the csv file: {self.csv_results}")