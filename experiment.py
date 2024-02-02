import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import ParameterGrid
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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
        self.train_metrics_exl = ['f1','acc_2off','qwk','spearman','amae','mmae','ms']
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

    @property
    def seed(self):
        return self.set_config.get('seed', 42)


    # build experiment name based on settings
    def build_exp_name(self, gridsearch):
        # parameters not to be used to generate the experiment name
        if not gridsearch:
            incl_params = ["nn_model", "nn_backbone", "epochs", "augmentation", "loss", "optimizer"]
            experiment_params = {key: value for key, value in self.settings.items() if key in incl_params}
        else:
            excl_params = ["folds", "augmentation", "metrics", "weight_decay", "momentum"]
            experiment_params = {key: value for key, value in self.settings.items() if key not in excl_params}

        # generate the experiment name based on the parameters
        exp_name = "_".join(str(value) for value in experiment_params.values())
        
        return f"seed{self.seed}_{exp_name}"
        #return exp_name


    # building settings of the experiment class
    def build(self, gridsearch):
        # build the experiment name based on the configuration extracted
        self.exp_name = self.build_exp_name(gridsearch) 
        
        # create the experiment results subdirectory
        self.exp_results_subdir = os.path.join(self.results_dir, self.exp_name)
        os.makedirs(self.exp_results_subdir, exist_ok=True)
        
        # add the experiment name in the results dictionary
        self.metrics_results['experiment'] = self.exp_name

        return self.exp_name


    # set the experiment state to the current fold
    def set_current_fold(self, current_fold, train_set, test_fold):
        self.current_fold = current_fold
        self.fold_train_set = train_set
        self.fold_test_set = test_fold

        # create the fold sub-directory of the experiment
        self.fold_subdir = os.path.join(self.exp_results_subdir, f'fold_{self.current_fold}/')
        os.makedirs(self.fold_subdir , exist_ok=True)

        # dict to store the MAE scores of the gridsearch holdouts
        self.splits_mae_scores = {}


    # extract patients from train and test folds and save them to a JSON file
    def from_fold_split_to_pats(self, train_folds, test_fold):
        # extract groups and labels for the train fold
        train_pats = np.unique([self.dataset.groups[movie] for movie in train_folds])
        test_pats = np.unique([self.dataset.groups[movie] for movie in test_fold])
        
        # save the split
        split_data = {"train_patients": train_pats.tolist(), "test_patients": test_pats.tolist()}
        split_save_path = os.path.join(self.fold_subdir, 'fold_patients_split.json')
        with open(split_save_path, 'w') as json_file:
            json.dump(split_data, json_file)

        return train_pats, test_pats


    # set the experiment state to the current HPV holdout (grid search phase)
    def set_current_hpv_holdout(self, hpv_curr_split, hpv_train, hpv_val):
        self.hpv_curr_split = hpv_curr_split
        self.hpv_train = list(hpv_train)
        self.hpv_val = list(hpv_val)
        
        # dict to store the best MAE scores for each parameter combination (for each train)
        self.mae_scores = {} 

        # create the folder for this grid search holdout in the current fold
        # self.hpv_holdout_dir = os.path.join(self.fold_subdir, f'holdout_hpv{self.hpv_curr_split}/')
        # os.makedirs(self.hpv_holdout_dir , exist_ok=True)
    

    # prepare the HPV datasets (train and val) and extract labels
    def prepare_hpv_sets(self):
        self.hpv_train, self.y_hpv_train = self.dataset.build_tfrecord_from_patients(self.hpv_train)
        self.hpv_val, self.y_hpv_val = self.dataset.build_tfrecord_from_patients(self.hpv_val)

    
    # compute space of hyperparameters by generating the parameters grid
    def get_hyperparameters_grid(self):
        # built by searching all the experiment's settings that are a list of values
        # and by using ParameterGrid() from sklearn to compute a grid of all combinations
        hps = self.settings.items()
        excl_keys = ['metrics']
        hps_comb = {key: value for key, value in hps if isinstance(value, list) and key not in excl_keys}
        
        return ParameterGrid(hps_comb)
    
    
    # evaluating the current hyperparameters in the grid search phase
    def evaluate_hyperparams(self, hyperparameters, epochs=10):
        # get global HP (valid parameters regardless of the model)
        batch_size = hyperparameters['batch_size']
        learning_rate = hyperparameters['learning_rate']
        
        if learning_rate == 'cdr':
            total_steps = epochs * (len(self.y_hpv_train) // batch_size)
            first_decay_steps = int(0.2 * total_steps)
            
            learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=1e-3,
                first_decay_steps=first_decay_steps,
                t_mul=1.6,
                m_mul=0.8,
                alpha=1e-4
            )
        
        # get fixed parameters from settings
        augmentation = self.settings['augmentation']
        
        # generate the train, (val) and test sets to feed the neural networks
        hpv_train = self.dataset.generate_tfrset(self.hpv_train, batch_size=batch_size, shuffle=True, augment=augmentation)
        hpv_val = self.dataset.generate_tfrset(self.hpv_val, batch_size=batch_size)
        
        # build the model with current parameters
        cv_model = self.build_model(hyperparameters=hyperparameters)

        # compile the model with the current LR
        self.compile_model(cv_model, learning_rate=learning_rate)
        
        # training the model for 10 epochs (default)
        hpv_history = self.cv_model_train(cv_model,
                                          train_ds=hpv_train,
                                          val_ds=hpv_val,
                                          y_train=self.y_hpv_train,
                                          y_val=self.y_hpv_val,
                                          epochs=epochs, 
                                          batch_size=batch_size)
        
        # get the smallest MAE in the validation set
        min_mae_score = min(hpv_history.history['val_mae'])

        # save the MAE associated with this combination of tested hyper-parameters
        self.mae_scores[str(hyperparameters)] = min_mae_score
        

    # compute the results of the grid search phase, save and extract the best parameters combo
    def compute_grid_search_results(self):
        # extract all unique parameter combinations
        param_combinations = set(param for split in self.splits_mae_scores.values() for param in split.keys())

        # compute averages for each parameter combination for all holdouts
        mean_values = {}
        for param in param_combinations:
            values = [split.get(param, 0) for split in self.splits_mae_scores.values()]
            mean_values[param] = np.mean(values)
        
        # find the combination with the lowest average
        best_params = min(mean_values, key=mean_values.get)
        min_mean = mean_values[best_params]

        # print the best combination with its MAE score
        print("\nBest hyperparameters combination:")
        print(f"{best_params}: {min_mean}\n")
        
        # save the grid search results to the current fold folder
        hpcv_file = os.path.join(self.fold_subdir, 'grid_search_results.txt')
        with open(hpcv_file, 'w') as file:            
            for param, mean in mean_values.items():
                file.write(f"{param}\t{mean}\n")
            file.write("\nBest hyperparameters combination:\n")
            file.write(f"{best_params}\t{min_mean}\n")

        # backward type conversion: from str -> dict
        return eval(best_params)


    # set the experiment state to the current HPT holdout (model training phase)
    def set_current_hpt_holdout(self, hpt_curr_split, hpt_train, hpt_val):
        self.hpt_curr_split = hpt_curr_split
        self.hpt_train = list(hpt_train)
        self.hpt_val = list(hpt_val)

        # create the folder for this training holdout in the current fold
        self.hpt_holdout_dir = os.path.join(self.fold_subdir, f'holdout_{self.hpt_curr_split}/')
        os.makedirs(self.hpt_holdout_dir , exist_ok=True)

        # create the weights folder inside of it
        hpt_holdout_weights_dir = os.path.join(self.hpt_holdout_dir, 'weights/')
        os.makedirs(hpt_holdout_weights_dir , exist_ok=True)

        # save the split in a JSON file locally
        split_data = {"train_patients": self.hpt_train, "val_patients": self.hpt_val}
        split_save_path = os.path.join(self.hpt_holdout_dir, f'hpt{self.hpt_curr_split}_patients_split.json')
        with open(split_save_path, 'w') as json_file:
            json.dump(split_data, json_file)


    # prepare the HPT datasets (train and val) and extract labels
    def prepare_hpt_sets(self):
        self.hpt_train, self.y_hpt_train = self.dataset.build_tfrecord_from_patients(self.hpt_train)
        self.hpt_val, self.y_hpt_val = self.dataset.build_tfrecord_from_patients(self.hpt_val)


    # training the model with the best parameters extracted from the grid search phase
    def hpt_train_network(self, best_hyperparameters):
        # gather the settings and best parameters
        epochs = self.settings['epochs']
        augmentation = self.settings['augmentation']
        batch_size = best_hyperparameters['batch_size']
        learning_rate = best_hyperparameters['learning_rate']
        
        # fix for reduce LR on plateau if using the scheduler
        rlop = True

        if learning_rate == 'cdr':
            total_steps = epochs * (len(self.y_hpt_train) // batch_size)
            first_decay_steps = int(0.2 * total_steps)
            
            learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=1e-3,
                first_decay_steps=first_decay_steps,
                t_mul=1.6,
                m_mul=0.8,
                alpha=1e-4
            )

            rlop = False

        # generate the train, (val) and test sets to feed the neural networks
        hpt_train = self.dataset.generate_tfrset(self.hpt_train, batch_size=batch_size, shuffle=True, augment=augmentation)
        hpt_val = self.dataset.generate_tfrset(self.hpt_val, batch_size=batch_size)
        
        # build the model with best hyperparameters
        hpt_model = self.build_model(hyperparameters=best_hyperparameters)
        
        # compile the model with the best learning rate
        self.compile_model(hpt_model, learning_rate=learning_rate)
        
        # training
        hpt_history = self.hpt_model_train(hpt_model,
                                           train_ds=hpt_train,
                                           val_ds=hpt_val,
                                           y_train=self.y_hpt_train,
                                           y_val=self.y_hpt_val,
                                           epochs=epochs, 
                                           batch_size=batch_size,
                                           gradcam_freq=3,
                                           rlop=rlop
                                           )
        
        return hpt_model, hpt_history


    # testing the network model with the current test fold 
    def hpt_test_network(self, test_pats, hpt_model, best_hyperparameters):
        batch_size = best_hyperparameters['batch_size']
        
        # prepare and generate the test set (with the patients in the current test fold)
        hpt_test, hpt_y_test = self.dataset.build_tfrecord_from_patients(list(test_pats))
        hpt_test = self.dataset.generate_tfrset(hpt_test, batch_size=batch_size)
        
        # evaluate the neural network 
        self.model_evaluate(hpt_model,
                            hpt_test,
                            hpt_y_test,
                            batch_size=batch_size
                            )

    
    # method to build a neural network model with specific parameters
    def build_model(self, hyperparameters):
        # get the network type: obd, clm, resnet18, cnn128, vgg16
        nn_model = self.settings['nn_model']
        
        # get the common parameters between models
        common_params = {
            'ds_img_size': self.ds_img_size,
            'ds_img_channels': self.ds_img_channels,
            'ds_num_classes': self.ds_num_classes,
            'nn_dropout': hyperparameters['dropout']
        }

        if nn_model == 'obd':
            net_object = NeuralNetwork(nn_backbone = self.settings['nn_backbone'],
                                       obd_hidden_size = hyperparameters['hidden_size'], 
                                       **common_params)
        elif nn_model == 'clm':
            net_object = NeuralNetwork(nn_backbone = self.settings['nn_backbone'],
                                       clm_link = hyperparameters['link_function'], 
                                       clm_use_tau = hyperparameters['use_tau'], 
                                       **common_params)
        else:
            net_object = NeuralNetwork(**common_params)

        # build the defined neural network model 
        model = net_object.build(nn_model)

        # auto-search the last convolutional layer of the model (uselful for GRAD-cams)
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                self.last_conv_layer = layer.name
                break
        
        print('◇ model built')

        return model
    

    # method to compile a neural network model with a specific LR
    def compile_model(self, model, learning_rate, summary=False):
        loss = self.settings['loss']
        metrics = self.settings['metrics']
        optimizer = self.settings['optimizer']
        
        # optimizer
        if optimizer.lower() == 'adam':
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
        metrics_t = Metrics(self.ds_num_classes, self.settings['nn_model'])
        train_metrics = [getattr(metrics_t, metric_name) for metric_name in metrics if metric_name not in self.train_metrics_exl]
        
        # compile
        model.compile(optimizer=optimizer, loss=loss, metrics=train_metrics)

        if summary:
            print(model.summary())

        print('◇ model compiled')


    # method to train a neural network model in the grid search phase (HPV)
    def cv_model_train(self, model, train_ds, val_ds, y_train, y_val, epochs, batch_size):
        # clear session
        tf.keras.backend.clear_session()

        # calcolo peso classi
        class_weight = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight = dict(enumerate(np.round(class_weight, self.ds_num_classes)))

        # compute train and val steps per epoch
        train_steps_per_epoch = len(y_train) // batch_size
        val_steps_per_epoch = len(y_val) // batch_size
        
        # neural network fit
        history = model.fit(train_ds,
                            epochs=epochs,
                            steps_per_epoch=train_steps_per_epoch,
                            class_weight=class_weight,
                            validation_data=val_ds,
                            validation_steps=val_steps_per_epoch,
                            verbose=self.verbose,
                            max_queue_size=self.max_qsize,
                            workers=self.workers,
                            use_multiprocessing=False
                            )

        return history


    # method to train a neural network model in the training phase (HPT)
    def hpt_model_train(self, model, train_ds, val_ds, y_train, y_val, epochs, batch_size, gradcam_freq=0, rlop=True):
        # clear session
        tf.keras.backend.clear_session()
        
        # parameters
        ckpt_filename = os.path.join(self.hpt_holdout_dir, 'weights/', 'best_weights.h5')
        #ckpt_filename = os.path.join(self.exp_results_subdir, "weights/", f"{cvcs}_best_weights.h5")
        log_dir = f"logs/fit/{self.exp_name}_fold{self.current_fold}_holdout{self.hpt_curr_split}"
        #log_dir = os.path.join(self.hpt_holdout_dir, "logs/fit/")
        es_patience = 25 if rlop else 30
        
        # callbacks
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint = ModelCheckpoint(ckpt_filename, monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=self.verbose)
        early_stop = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=self.verbose) 
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, min_lr=1e-7, verbose=self.verbose) if rlop else None
        gradcam = GradCAMCallback(model, self, val_data=val_ds, freq=gradcam_freq) if gradcam_freq > 0 else None

        # build callbacks list
        callbacks = [tensorboard, checkpoint, early_stop, reduce_lr, gradcam]
        callbacks = [callback for callback in callbacks if callback is not None]

        # calcolo peso classi
        class_weight = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight = dict(enumerate(np.round(class_weight, self.ds_num_classes)))

        # compute train and val steps per epoch
        train_steps_per_epoch = len(y_train) // batch_size
        val_steps_per_epoch = len(y_val) // batch_size
        
        # neural network fit   
        history = model.fit(train_ds,
                            epochs=epochs,
                            steps_per_epoch=train_steps_per_epoch,
                            class_weight=class_weight,
                            validation_data=val_ds,
                            validation_steps=val_steps_per_epoch,
                            callbacks=callbacks,
                            verbose=self.verbose,
                            max_queue_size=self.max_qsize,
                            workers=self.workers,
                            use_multiprocessing=False
                            )

        return history
    

    # method to save the training charts
    def nn_train_graphs(self, history):
        # get the output mode
        display, save = self.output_mode

        # create a figure with two subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # plot loss functions
        for loss in history.history.keys():
            if loss.endswith('loss'):
                label = loss
                linestyle = '--' if loss.startswith('val_') else '-'
                ax1.plot(history.history[loss], label=label, linestyle=linestyle)
        
        ax1.legend()
        ax1.set_xlabel('epoch')
        ax1.set_ylabel(f'{self.settings["loss"]}')
        ax1.set_title(f'Loss - {self.exp_name}_{self.current_fold}_{self.hpt_curr_split}')
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
        ax2.set_title(f'Metrics - {self.exp_name}_{self.current_fold}_{self.hpt_curr_split}')
        ax2.grid()

        # save the training charts
        if save:
            train_graphs_path = os.path.join(self.hpt_holdout_dir, "training_plots.png")
            plt.savefig(train_graphs_path, bbox_inches='tight', pad_inches=0.2)
        
        # Show the figure
        if display:
            plt.show()
        
        plt.close()

        print('◇ training graphs saved')


    # method to evaluate the trained model on the fold set
    def model_evaluate(self, model, X_test, y_test, batch_size):
        # load the best weights
        best_weights_file = os.path.join(self.hpt_holdout_dir, 'weights/', 'best_weights.h5')
        try:
            model.load_weights(best_weights_file)
        except Exception as e:
            raise Exception('error while loading best weights file: ', e)
        
        # compute test steps per epoch
        test_steps_per_epoch = -(-len(y_test) // batch_size)

        # model evaluation, get the predictions by running the model inference
        y_test_pred = model.predict(X_test,
                                    steps=test_steps_per_epoch,
                                    verbose=self.verbose,
                                    max_queue_size=self.max_qsize,
                                    workers=self.workers,
                                    use_multiprocessing=False
                                    )
        
        # save the ground truth and predictions in a JSON file locally
        predictions_to_save = {"y_test": y_test.tolist(), "y_test_pred": y_test_pred.tolist()}
        predictions_save_path = os.path.join(self.hpt_holdout_dir, 'predictions.json')
        with open(predictions_save_path, 'w') as json_file:
            json.dump(predictions_to_save, json_file)
        
        # compute evaluation metrics
        metrics_e = Metrics(self.ds_num_classes, self.settings['nn_model'])
        eval_metrics = [(getattr(metrics_e, metric_name), metric_name) for metric_name in self.settings['metrics']]
        
        for metric, metric_name  in eval_metrics:
            result = metric(y_test, y_test_pred)
            result = np.round(result, 4)
            self.metrics_results[metric_name] = result
        
        # test set confusion matrix
        display, save = self.output_mode
        cfmat_fig = metrics_e.confusion_matrix(y_test, y_test_pred, show=display)
        if save:
            cfmat_fig_path = os.path.join(self.hpt_holdout_dir, 'confusion_matrix.png')
            cfmat_fig.savefig(cfmat_fig_path, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        
        # ROCs curves
        roc_curves = metrics_e.rocs_per_class(y_test, y_test_pred)
        if save:
            roc_curves_fig_path = os.path.join(self.hpt_holdout_dir, 'roc_curves_per_class.png')
            roc_curves.savefig(roc_curves_fig_path, bbox_inches='tight', pad_inches=0.2)
        plt.close()

        # save results on the csv   
        # add the experiment name in the results dictionary
        self.metrics_results['experiment'] = f"{self.exp_name}_fold{self.current_fold}_split{self.hpt_curr_split}"   
        values_columns = [self.metrics_results.get(column, '-') for column in self.csv_columns]
        with open(self.csv_results_path, mode='a', encoding='UTF-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(values_columns)