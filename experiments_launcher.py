import os
import shutil
import argparse
import json
import sys
from rich.console import Console
from experiment import Experiment
from keras import backend as K
from logger import Logger

def load_experiments_json(experiments_file):    
    try:
        with open(experiments_file, 'rb') as file:
            exps = json.load(file)
        return exps, len(exps)
    except FileNotFoundError:
        Console().print(f"experiments json file '{experiments_file}' not found, exit.")
        sys.exit(1)
    except json.JSONDecodeError:
        Console().print(f"error reading experiments json file '{experiments_file}', exit.")
        sys.exit(1)

def check_run_status(checkpoint_file):
    try:
        with open(checkpoint_file, 'r') as file:
            return len(file.readlines())
    except FileNotFoundError:
        return 0

def find_common_settings(experiments_list):
    common_params = {}
    
    # failsafe
    if not experiments_list:
        return common_params

    # get settings from the first experiment to use as baseline
    first_experiment = experiments_list[0]
    
    # check the setting common to all experiments and save them
    for key, value in first_experiment.items():
        if all(exp.get(key) == value for exp in experiments_list):
            common_params[key] = value

    return common_params

def main():
    parser = argparse.ArgumentParser(description="LUS Ordinal Classification")
    parser.add_argument("--exps_json", type=str, required=True, help="json file containing the experiments to be performed")
    parser.add_argument("--dataset", type=str, required=True, help="dataset in HDF5 format")
    parser.add_argument("--ds_map", type=str, required=True, help="pickle containing the dataset frames indexes mapping")
    parser.add_argument("--ds_split", type=str, required=True, help="pickle containing the dataset infos needed for the splitting")
    parser.add_argument("--results_dir", type=str, default="results/", help="directory used to store the results")
    parser.add_argument("--buffer_size", type=int, default=100, help="buffer size used to shuffle the training dataset")
    parser.add_argument("--workers", type=int, default=1, help="processes employed when using process-based threading")
    parser.add_argument("--verbose", type=int, default=1, help="verbose intensity for the whole experiment")
    parser.add_argument("--seed", type=int, default=42, help="seed used to initialize the random number generator")
    args = parser.parse_args()
    
    # load the experiments JSON file
    experiments, tot_exps = load_experiments_json(args.exps_json)

    # useful classes initialization
    logger = Logger(args, experiments)
    console = Console()

    # extract common settings, if any, across experiments
    common_settings = find_common_settings(experiments)
    same_ds_splitting = all(key in common_settings for key in ['ds_split_ratio', 'ds_split_by', 'ds_trim', 'ds_us'])
    
    # initialization of the class which represents a particular experiment with a defined set of parameters
    experiment = Experiment(exps_json_path = args.exps_json,
                            dataset_path = args.dataset,
                            ds_map_pkl = args.ds_map,
                            ds_split_pkl = args.ds_split,
                            results_dir = args.results_dir,
                            workers = args.workers,
                            shuffle_buffer_size = args.buffer_size,
                            verbose = args.verbose,
                            random_state = args.seed)
    
    # building the class
    experiment.build()
    
    # provide the current experiment at the logger 
    logger.update_experiment(experiment)

    # print sessions settings
    logger.print_settings()

    # check the run status
    if check_run_status('run_checkpoint.txt') > 0:
        console.log(f"[bold cyan]resuming run[/bold cyan] from last experiment\r")
    
    # if the settings regarding the dataset are equal for each experiment, perform them once for all [MHA <3]
    if os.path.exists('splitdata.pkl'):
        experiment.load_dataset_splitted()
        console.log(f"[bold cyan]dataset splitting loaded[/bold cyan] (from pickle file)\r")
    elif not os.path.exists('splitdata.pkl') and same_ds_splitting:
        experiment.split_dataset(exps_common_settings=common_settings)
    
    if same_ds_splitting:
        experiment.compute_class_weight()
        experiment.generate_split_charts()
        logger.print_ds_splitting()
        
    with console.status("[bold green]Starting...\n") as status:
        for exp_idx, _ in enumerate(experiments):
            # first of all clear the session
            K.clear_session()
            
            # load the experiment settings
            exp_name = experiment.load_exp_settings(exp_idx)    
            
            # check the experiment checkpoint
            if experiment.check_exps_ckp(exp_name):
                console.log(f"skipping experiment [bold cyan]{exp_name}[/bold cyan] ([bold yellow]previously completed[/bold yellow]) [{exp_idx + 1}/{tot_exps}]\r")
                continue
            
            # perform the dataset splitting and computing the class weight if not done before
            if not same_ds_splitting:
                experiment.split_dataset()
                experiment.compute_class_weight()
                experiment.generate_split_charts(per_exp=True)
                logger.print_ds_splitting()
            
            # generate the train, val and test sets based on the batch size
            experiment.generate_sets()
            
            # build the neural network model using the current experiment settings
            model = experiment.nn_model_build()

            # compile the neural network model using the current experiment settings
            experiment.nn_model_compile(model, summary=False)
            
            # logging the current experiment's model
            logger.update_experiment(experiment)
            #logger.print_model_params()
            
            # update the console status
            mess = f"[bold green]Training model of experiment '{exp_name}' [{exp_idx+1}/{tot_exps}]\r"
            status.update(mess)
            
            # train the neural network model
            history = experiment.nn_model_train(model, status_bar=(status, mess))
            
            # plot training graphs
            experiment.nn_train_graphs(history)
            
            # update the console status
            status.update(f"[bold green]Evaluating model of experiment '{exp_name}' [{exp_idx+1}/{tot_exps}]\n")

            # evaluating the neural network model
            experiment.nn_model_evaluate(model)

            # logging the end of the current experiment
            console.log(f"experiment [bold cyan]{exp_name}[/bold cyan] [bold green]completed[/bold green] [{exp_idx + 1}/{tot_exps}]\r")

        # remove run checkpoint file
        if os.path.exists('run_checkpoint.txt'):
            os.remove('run_checkpoint.txt')
            console.log(f"run completed, [bold cyan]checkpoint file removed[/bold cyan]\r")

if __name__ == "__main__":
    main()
    os._exit(0)