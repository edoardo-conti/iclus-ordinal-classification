import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from keras import backend as K
from keras.callbacks import TensorBoard, BackupAndRestore, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utilities import plot_charts
from network.model import NeuralNetwork
from network.losses import make_cost_matrix, qwk_loss, ordinal_distance_loss
from network.metrics import Metrics
from network.callbacks import GradCAMCallback

class Experiment:
    def __init__(self, 
                 settings:dict,
                 set_config:dict):
        self.settings = settings
        self.set_config = set_config
        
        # ======= to compute =======
        self.exp_name = None
        self.exp_results_subdir = None       
        
        # ======= train set ========
        self.x_train = None
        self.y_train = None
        self.train_class_weights = None
        # ===== validation set =====
        self.x_val = None
        self.y_val = None
        # ======= test set =========
        self.x_test = None
        self.y_test = None
        
        # ==== neural network ======
        self.last_conv_layer = None
        self.train_metrics_exl = ['ms', 'qwk']
        self.metrics_results = {}
    

    @property
    def results_dir(self):
        return self.set_config.get('results_dir', 'results/')
    
    @property
    def dataset(self):
        return self.set_config.get('dataset', None)
    
    @property
    def output_mode(self):
        return self.set_config.get('output_mode', (0, 1))

    @property
    def ds_img_size(self):
        return self.set_config.get('ds_img_size', 224)

    @property
    def ds_img_channels(self):
        return self.set_config.get('ds_img_channels', 3)
    
    @property
    def ds_num_classes(self):
        return self.set_config.get('ds_num_classes', 4)
    
    @property
    def verbose(self):
        return self.set_config.get('verbose', 0)
    
    @property
    def max_qsize(self):
        return self.set_config.get('max_qsize', 100)
    
    @property
    def workers(self):
        return self.set_config.get('workers', 1)
    
    @property
    def csv_columns(self):
        return self.set_config.get('csv_columns', [])
    
    @property
    def csv_results_path(self):
        return self.set_config.get('csv_results_path', 'results/results.csv')


    def build_exp_name(self):
        # parameters not to be used to generate the experiment name
        excl_params = ["ds_split_ratio", "metrics"]
        experiment_params = {key: value for key, value in self.settings.items() if key not in excl_params}

        # generate the experiment name based on the parameters
        exp_name = "_".join(str(value) for value in experiment_params.values())

        return exp_name


    def build(self):
        # build the experiment name based on the configuration extracted
        self.exp_name = self.build_exp_name()
        
        # create the experiment results subdirectory
        self.exp_results_subdir = os.path.join(self.results_dir, self.exp_name)
        os.makedirs(self.exp_results_subdir, exist_ok=True)

        # create the weights subdirectory
        weights_path = os.path.join(self.exp_results_subdir, 'weights/') 
        os.makedirs(weights_path, exist_ok=True)

        # add the experiment name in the results dictionary
        self.metrics_results['experiment'] = self.exp_name


    def split_dataset(self):
        # gather the needed experiment settings
        split_ratio, m_rus = [self.settings[key] for key in ['ds_split_ratio', 'ds_mrus']]
        
        # split dataset
        self.dataset.split_dataset(split_ratio)

        # prepare sets
        self.x_train, self.y_train = self.dataset.prepare_tfrset('train', random_under_msampler=m_rus)
        self.x_val, self.y_val = self.dataset.prepare_tfrset('val', random_under_msampler=m_rus)
        self.x_test, self.y_test = self.dataset.prepare_tfrset('test', random_under_msampler=m_rus)
        
        # generate sets
        # TODO: forse si può ottimizzare e ridurre l'overhead qui
        # gather the needed settings and data
        batch_size = self.settings['nn_batch_size']
        augmentation = self.settings['augmentation']

        # create the train, (val) and test sets to feed the neural networks
        self.x_train = self.dataset.generate_tfrset(self.x_train, 
                                                    batch_size=batch_size, 
                                                    shuffle=True,
                                                    augment=augmentation)
        self.x_val = self.dataset.generate_tfrset(self.x_val, batch_size=batch_size) 
        self.x_test = self.dataset.generate_tfrset(self.x_test, batch_size=batch_size)

        print('◇ dataset splitted')
        

    def compute_class_weight(self):
        # calculate class balance using 'compute_class_weight'
        train_class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        train_class_weights = np.round(train_class_weights, 4)

        self.train_class_weights = dict(enumerate(train_class_weights))

        print('◇ class weights computed')


    def generate_split_charts(self, charts=None):
        # default graphs
        charts = charts or ["pdistr", "lsdistr_pie"]
        
        # get the output mode
        display, save = self.output_mode
        
        plot_charts(self, charts, display, save, self.exp_results_subdir)

        print('◇ split charts generated')


    def plot_set_batches(self, set='train', num_batches=10):
        # gather the needed settings and data
        batch_size = self.settings['nn_batch_size']
        # color map
        class_colors = {0: 'green', 1: 'darkblue', 2: 'darkorange', 3: 'darkred'}
        
        sets = {'train': self.x_train, 'val': self.x_val}
        selected_set = sets.get(set, self.x_test)

        for batch in selected_set.take(num_batches): 
            #_, axes = plt.subplots(2, 8, figsize=(20, 6))
            _, axes = plt.subplots(batch_size // 8, 8, figsize=(20, 3 * (batch_size // 8)))
            
            frames, labels = batch
            
            for i, (frame, label) in enumerate(zip(frames, labels)):        
                # Stampa l'immagine
                #axes[i % 2, i // 2].imshow(frame)
                axes[i // 8, i % 8].imshow(frame)

                # imposta colore e stampa etichetta
                color = class_colors.get(np.argmax(label), 'black')
                #axes[i % 2, i // 2].set_title(f'Target: {label}', color=color)
                axes[i // 8, i % 8].set_title(f'Target: {label}', color=color)

                # Nascondi gli assi
                #axes[i % 2, i // 2].axis('off')
                axes[i // 8, i % 8].axis('off')

            plt.tight_layout()
            plt.show()
            plt.close()


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
                                       obd_hidden_size = self.settings['obd_hidden_size'], 
                                       **common_params)
        elif net_type == 'clm':
            net_object = NeuralNetwork(nn_backbone = self.settings['nn_backbone'],
                                       clm_link = self.settings['clm_link'], 
                                       clm_use_tau = self.settings['clm_use_tau'], 
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
        
        print('◇ model built')

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

        print('◇ model compiled')


    def nn_model_train(self, model, gradcam_freq=5, status_bar=None):
        # parameters
        epochs = self.settings['nn_epochs']
        batch_size = self.settings['nn_batch_size']
        ckpt_filename = os.path.join(self.exp_results_subdir, "weights/", "best_weights.h5")

        # callbacks
        tensorboard = TensorBoard(log_dir=f"logs/fit/{self.exp_name}", histogram_freq=1)
        backup = BackupAndRestore(backup_dir="backup/")
        checkpoint = ModelCheckpoint(ckpt_filename, monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=self.verbose)
        early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=self.verbose)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=self.verbose)
        gradcam = GradCAMCallback(model, self, freq=gradcam_freq) if gradcam_freq > 0 else None

        # build callbacks list
        callbacks = [tensorboard, backup, checkpoint, early_stop, reduce_lr, gradcam]
        callbacks = [callback for callback in callbacks if callback is not None]
        
        # compute train and val steps per epoch
        train_steps_per_epoch = self.dataset.frame_counts['train'] // batch_size
        val_steps_per_epoch = self.dataset.frame_counts['val'] // batch_size

        # neural network fit
        history = model.fit(self.x_train,
                            epochs=epochs,
                            steps_per_epoch=train_steps_per_epoch,
                            class_weight=self.train_class_weights,
                            validation_data=self.x_val,
                            validation_steps=val_steps_per_epoch,
                            callbacks=callbacks,
                            verbose=self.verbose,
                            max_queue_size=self.max_qsize,
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

        print('◇ training graphs saved')


    def nn_model_evaluate(self, model, weights=None, load_best_weights=True):
        if weights is not None:
             # if specified, load the weights passed as argument (priority)
            model.load_weights(weights)
        elif load_best_weights:
            # load the best weights
            best_weights_file = os.path.join(self.exp_results_subdir, "weights", "best_weights.h5")
            try:
                model.load_weights(best_weights_file)
            except Exception as e:
                raise Exception('error while loading best weights file: ', e)
        
        # get the batch size
        batch_size = self.settings['nn_batch_size']
        
        # compute test steps per epoch
        test_steps_per_epoch = -(-self.dataset.frame_counts['test'] // batch_size)

        # model evaluation, get the predictions by running the model inference
        y_test_pred = model.predict(self.x_test,
                                    steps=test_steps_per_epoch,
                                    verbose=self.verbose,
                                    max_queue_size=self.max_qsize,
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
        