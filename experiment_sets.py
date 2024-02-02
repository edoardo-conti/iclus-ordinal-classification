import os
import sys
import shutil
import csv
import json
import datetime
from typing import Tuple
import tensorflow as tf
from keras import mixed_precision
from dataset import DatasetHandler

class ExperimentSets:
    def __init__(self, 
                 exps_json_path:str,
                 dataset_dir:str,
                 results_dir:str,
                 mp_xla:Tuple[int, int] = (0, 0),
                 workers:int = 1,
                 shuffle_bsize:int = 100,
                 max_qsize:int = 100,
                 verbose:int = 1,
                 output_mode:Tuple[int, int] = (0, 1),
                 seed:int = 42):
        self.exps_json_path = exps_json_path
        self.dataset_dir = dataset_dir
        self.results_dir = results_dir
        self.mp_xla = mp_xla
        self.workers = workers
        self.shuffle_bsize = shuffle_bsize
        self.max_qsize = max_qsize
        self.verbose = verbose
        self.output_mode = output_mode
        self.seed = seed
        
        # dataset attributes
        self.ds_img_size = 224
        self.ds_img_channels = 3
        self.ds_num_classes = 4

        # params to be computed
        self.host_hw = {}
        self.exps = None
        self.tot_exps = 0
        self.logs_path = None
        self.dataset = None
        self.csv_results_path = None
        self.csv_columns = []
                

    def build(self):
        # setting os environment
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        
        # executing a series of initialization functions for the run
        self.check_gpu_availability()
        self.check_mixedp_and_xla()
        self.load_experiments()
        self.load_dataset()
        self.init_results()
        self.init_logs()

        return self.export_set_config()


    def check_gpu_availability(self):
        cpu_device = tf.config.list_physical_devices('CPU')[0]
        gpu_devices = tf.config.list_physical_devices('GPU')
        
        if gpu_devices:
            device = gpu_devices[0]
            details = tf.config.experimental.get_device_details(device)
            gpu_name = details.get('device_name', 'Unknown')
            gpu_capability = float(".".join(map(str, details.get('compute_capability', '0'))))
            self.host_hw.update({'gpu_name': gpu_name, 'gpu_capability': gpu_capability})
            print(f'✔ gpu {gpu_name} available (compute capability: {gpu_capability})')
        else:
            device = cpu_device
        
        self.host_hw['device'] = device
    

    def check_mixedp_and_xla(self):
        try:
            device_type = self.host_hw['device'].device_type
            gpu_capability = self.host_hw['gpu_capability']

            if device_type == 'GPU' and gpu_capability >= 7.0:
                mp, xla = self.mp_xla
                if mp:
                    # Mixed Precision 
                    policy = mixed_precision.Policy("mixed_float16")
                    mixed_precision.set_global_policy(policy)
                    print(f'✔ mixed precision on')

                if xla:
                    # XLA (Accelerated Linear Algebra)
                    tf.config.optimizer.set_jit(True)
                    print(f'✔ XLA on')
            else:
                print(f'✘ mixed precision and XLA off')
        except Exception as e:
            print(f'✘ error enabling Mixed Precision and XLA: {e}')


    def load_experiments(self):
        try:
            with open(self.exps_json_path, 'rb') as file:
                self.exps = json.load(file)
                self.tot_exps = len(self.exps)
            print(f'✔ {self.tot_exps} experiments loaded')
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"✘ error loading experiments json file '{self.exps_json_path}': {e}, exit.")
            sys.exit(1)
    

    def load_dataset(self):
        self.dataset = DatasetHandler(dataset_dir = self.dataset_dir, 
                                      shuffle_bsize = self.shuffle_bsize,
                                      seed = self.seed)
        self.dataset.build()
        print('✔ dataset loaded')


    def init_results(self):
        # declare the CSV file path and columns
        self.csv_results_path = os.path.join(self.results_dir, "results.csv")
        self.csv_columns = ["experiment", "ccr", "f1", "acc_1off", "acc_2off", "qwk", "spearman", "mae", "amae", "mmae", "rmse", "ms"]

        # TODO: before doing anything ask for a confirmation
        # confirmation = input("Do you want to delete existing run files? (y/[N]): ").lower()
        confirmation = 'y'
        
        # experiments checkpoint file 
        if confirmation == 'y':
            # clean the previous results and re-make the directory
            if os.path.exists(self.results_dir):
                shutil.rmtree(self.results_dir)
            os.makedirs(self.results_dir)
            
            # remove previous logs/ directory
            if os.path.exists('logs/'):
                shutil.rmtree('logs/')

            # create the CSV file and write the header
            with open(self.csv_results_path, mode='w', encoding='UTF-8', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(self.csv_columns)
        
        print('✔ results initialized\n')

    
    def init_logs(self):
        # get the logs file path and current date/time
        self.logs_path = os.path.join(self.results_dir, 'experiments_logs.txt')
        timelog = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # log the start of the run
        with open(self.logs_path, 'w') as log_file: log_file.write(f"{timelog} - Start\n")

    
    def export_set_config(self):
        params_to_exclude = ['exps_json_path', 'dataset_dir', 'mp_xla', 
                             'shuffle_bsize', 'host_hw', 'exps', 'tot_exps']
        config = {key: value for key, value in self.__dict__.items() if key not in params_to_exclude}

        return config
