import argparse
from keras import backend as K
from experiment import Experiment
from experiment_sets import ExperimentSets

def main():
    parser = argparse.ArgumentParser(description="LUS Ordinal Classification")
    parser.add_argument("--exps_json", type=str, required=True, help="json file containing the experiments to be performed")
    parser.add_argument("--dataset", type=str, required=True, help="dataset folder")
    parser.add_argument("--results_dir", type=str, default="results/", help="directory used to store the results")
    parser.add_argument("--eval_weights", type=str, help="evaluate the model of the experiments using the provided weights")
    parser.add_argument("--mixedp", action='store_true', help="enable the mixed precision 'float_16/32'")
    parser.add_argument("--xla", action='store_true', help="enable the XLA (Accelerated Linear Algebra)")
    parser.add_argument("--workers", type=int, default=1, help="processes employed when using process-based threading")
    parser.add_argument("--shuffle_bsize", type=int, default=100, help="buffer size units used to shuffle the training dataset")
    parser.add_argument("--max_qsize", type=int, default=100, help="maximum size for the generator queue")
    parser.add_argument("--verbose", type=int, default=1, help="verbose intensity for the whole experiment")
    parser.add_argument("--seed", type=int, default=42, help="seed used to initialize the random number generator")
    args = parser.parse_args()
    
    print("\n★★★★★★★★★★★★★★★★★★★★★★★★ START ★★★★★★★★★★★★★★★★★★★★★★★★\n")

    # initialization of the set of experiments class with global settings
    experiments_set = ExperimentSets(exps_json_path = args.exps_json,
                                     dataset_dir = args.dataset,
                                     results_dir = args.results_dir,
                                     mp_xla = (args.mixedp, args.xla),
                                     workers = args.workers,
                                     shuffle_bsize = args.shuffle_bsize,
                                     max_qsize = args.max_qsize,
                                     verbose = args.verbose,
                                     random_state = args.seed)
    
    # building the class
    experiments_set.build()
    
    for exp_idx, exp_settings in enumerate(experiments_set.exps):
        # initialization of the experiment class with a defined set of parameters
        # TODO: migliorare il passaggio di parametri (magari scrivere metodo dedicato?)
        experiment = Experiment(settings=exp_settings, set_config=experiments_set.__dict__)
        
        # build the experiment 
        experiment.build()

        # get the ascii representation of the experiment's id
        print(f"\n◆ experiment {experiment.exp_name} [{exp_idx+1}/{experiments_set.tot_exps}] ...\n")

        # perform the dataset splitting, computing the class weight and generating the charts
        experiment.split_dataset()
        experiment.compute_class_weight()
        experiment.generate_split_charts()
        
        # build the neural network model using the current experiment settings
        model = experiment.nn_model_build()

        # compile the neural network model using the current experiment settings
        experiment.nn_model_compile(model, summary=False)

        print(f"\n☉ start training ...\n")

        # train the neural network model
        history = experiment.nn_model_train(model)

        print(f"\n☉ training completed\n")

        # plot training graphs
        experiment.nn_train_graphs(history)

        print(f"\n☉ start evaluating ...\n")

        # evaluating the neural network model
        experiment.nn_model_evaluate(model, weights=args.eval_weights)
        
        print(f"\n☉ evaluation of completed")
        
        # clear the session
        K.clear_session()

    print("\n★★★★★★★★★★★★ END ★★★★★★★★★★★★\n")

if __name__ == "__main__":
    main()