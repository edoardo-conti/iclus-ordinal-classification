from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table

class Logger:
    def __init__(self, args, exps):
        self.args = args
        self.exps = exps

        self.console = Console()

        self.exp = None

    def check_exp(self):
        if self.exp is None:
            raise Exception('experiment not loaded yet.')

    def update_experiment(self, experiment):
        self.exp = experiment

    def print_settings(self):
        self.check_exp()

        __exps_str = f"Experiments identified: {len(self.exps)} (from file: [magenta]{self.args.exps_json}[/magenta])\n"
        __results_str = f"Results directory: [magenta]{self.args.results_dir}[/magenta]\n"
        __seed_str = f"Global random seed: {self.args.seed}\n"
        __workers_str = f"Workers: {self.args.workers}\n"
        __hw_accel_str = f"GPU acceleration: {'[bold green]available' if self.exp.hw_accel else '[bold red]not available'}"
        __settings_panel_content = __exps_str + __results_str + __seed_str + __workers_str + __hw_accel_str
        __settings_panel = Panel(__settings_panel_content, title='[bold]Settings[/bold]', highlight=True)
        
        print(__settings_panel)
    
    def print_ds_splitting(self):
        self.check_exp()

        train_f_count = len(self.exp.idxs_train)
        val_f_count = len(self.exp.idxs_val)
        test_f_count = len(self.exp.idxs_test)
        ds_f_count = len(self.exp.dataset)
        tot_f_count = train_f_count + val_f_count + test_f_count

        #ds_us = self.exp.settings['ds_us']

        train_f_p = round((train_f_count / tot_f_count) * 100)
        val_f_p = round((val_f_count / tot_f_count) * 100)
        test_f_p = 100 - (train_f_p + val_f_p)
        
        __ds_str = f"Dataset: {self.exp.dataset.total_videos} videos and {self.exp.dataset.total_frames} frames (from file: [magenta]{self.args.dataset}[/magenta])\n"
        # __split_us_str = ""
        # if ds_us:
        #     if '2' in ds_us:
        #         __split_us_str = f"Dataset undersampling: [bold cyan]ON[/bold cyan] (using the 2nd minority class)\n"
        #     else:
        #         __split_us_str = f"Dataset undersampling: [bold cyan]ON[/bold cyan] (using the minority class)\n"
        __split_trim_str = "Dataset trimming: [bold cyan]OFF[/bold cyan] (using the [bold cyan]100%[/bold cyan] of the dataset)\n\n"
        if ds_f_count != tot_f_count:
            # dataset trimming
            red_perc = round(tot_f_count * 100 / ds_f_count)
            __split_trim_str = f"Dataset trimming: [bold cyan]ON[/bold cyan] (using the [bold cyan]{red_perc}%[/bold cyan] of the dataset)\n\n"
            
        __split_train_str = f"Training set\t= {train_f_count} frames ([bold cyan]{train_f_p}%[/bold cyan])\n"
        __split_val_str = f"Validation set\t= {val_f_count} frames ([bold cyan]{val_f_p}%[/bold cyan])\n"
        __split_test_str = f"Test set\t= {test_f_count} frames ([bold cyan]{test_f_p}%[/bold cyan])\n\n"
        __split_cw_str = f"Training set class weights: {self.exp.train_class_weights}"
        # __split_panel_content = __ds_str + __split_us_str + __split_trim_str + __split_train_str + __split_val_str + __split_test_str + __split_cw_str
        __split_panel_content = __ds_str + __split_trim_str + __split_train_str + __split_val_str + __split_test_str + __split_cw_str

        __split_panel = Panel(__split_panel_content, title='[bold]Dataset[/bold]', highlight=True)
        
        print(__split_panel)
    
    def print_model_params(self):
        self.check_exp()
        
        # ~~~ building model params ~~~
        net_type = self.exp.settings['nn_type']
        dropout = self.exp.settings['nn_dropout']

        if net_type == 'obd':
            net_backbone = self.exp.settings['nn_backbone']
            hidden_size = self.exp.settings['obd_hidden_size']

            print(f"'{net_type}' neural network built with:")
            print(f"\tbackbone -> {net_backbone}")
            print(f"\tdropout -> {dropout}")
            print(f"\tobd_hidden_size -> {hidden_size}\n")
        elif net_type == 'clm':
            net_backbone = self.exp.settings['nn_backbone']
            clm_link_function = self.exp.settings['clm_link_function']
            clm_use_tau = self.exp.settings['clm_use_tau']

            print(f"'{net_type}' neural network built with:")
            print(f"\tbackbone -> {net_backbone}")
            print(f"\tdropout -> {dropout}")
            print(f"\tclm_link_function -> {clm_link_function}")
            print(f"\tclm_use_tau -> {clm_use_tau}\n")
        else:
            print(f"'{net_type}' neural network built with:")
            print(f"\tdropout -> {dropout}\n")  

        print(f"last model's convolutional layer extracted: {self.exp.last_conv_layer} (needed by Grad-CAM)\n")

        # ~~~ compiling model params ~~~
        print("neural network model compiled with:")
        print(f"\tloss -> {self.exp.settings['loss']}")
        print(f"\toptimizer -> {self.exp.settings['optimizer']}")
        print(f"\tlearning rate -> {self.exp.settings['learning_rate']}\n")

       