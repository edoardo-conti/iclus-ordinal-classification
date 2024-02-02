import argparse
import utilities
from experiment_sets import ExperimentSets
from experiment import Experiment

# import numpy as np
# from sklearn.utils.class_weight import compute_class_weight

def main():
    parser = argparse.ArgumentParser(description="LUS Ordinal Classification")
    parser.add_argument("--exps_json", type=str, required=True, help="json file containing the experiments to be performed")
    parser.add_argument("--dataset", type=str, required=True, help="dataset folder")
    parser.add_argument("--results_dir", type=str, default="results/", help="directory used to store the results")
    parser.add_argument("--hpv_splits", type=int, default=3, help="number of splits for grid searching phase")
    parser.add_argument("--hpt_splits", type=int, default=3, help="number of splits to train models with")
    parser.add_argument("--no_gridsearch", action='store_true', help="avoid grid searching the best parameters")
    parser.add_argument("--mixedp", action='store_true', help="enable the mixed precision 'float_16/32'")
    parser.add_argument("--xla", action='store_true', help="enable the XLA (Accelerated Linear Algebra)")
    parser.add_argument("--workers", type=int, default=1, help="processes employed when using process-based threading")
    parser.add_argument("--shuffle_bsize", type=int, default=300, help="buffer size units used to shuffle the training dataset")
    parser.add_argument("--max_qsize", type=int, default=300, help="maximum size for the generator queue")
    parser.add_argument("--verbose", type=int, default=1, help="verbose intensity for the whole experiment")
    parser.add_argument("--seed", type=int, default=42, help="seed used to reproducibility")
    args = parser.parse_args()
    
    print("\n★★★★★★★★★★★★★★★★★★★★★★★★★★★★ START ★★★★★★★★★★★★★★★★★★★★★★★★★★★★\n")
    
    # initialization of the set of experiments class with global settings
    experiments_set = ExperimentSets(exps_json_path = args.exps_json,
                                     dataset_dir = args.dataset,
                                     results_dir = args.results_dir,
                                     mp_xla = (args.mixedp, args.xla),
                                     workers = args.workers,
                                     shuffle_bsize = args.shuffle_bsize,
                                     max_qsize = args.max_qsize,
                                     verbose = args.verbose,
                                     seed = args.seed)
    
    # building the class
    set_config = experiments_set.build()
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EXPERIMENTS LOOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for exp_id, exp_settings in enumerate(experiments_set.exps, start=1):
        # initialization of the experiment class with a defined configuration
        experiment = Experiment(settings=exp_settings, set_config=set_config)
        
        # build the experiment, it returns the experiment name
        exp_name = experiment.build(args.no_gridsearch)
        
        # log and print the builded experiment name with the experiments counter
        message = f"◆ experiment {exp_name} loaded [{exp_id}/{experiments_set.tot_exps}]"
        utilities.log_this(experiments_set.logs_path, message)

        # perform the Stratified Group K-Fold Cross-Validation 
        exp_ds = experiment.dataset
        num_folds = exp_ds.sgkfold(num_folds=experiment.settings['folds'], shuffle_folds=True)

        # TODO: da rimuovere
        # class_weights_collection = []

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ K-FOLDING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for curr_fold, (train_folds, test_fold) in enumerate(exp_ds.folds, start=1):
            print(f"\n~~~~~~~~~~~~~~~ FOLD {curr_fold}/{num_folds} ~~~~~~~~~~~~~~~")
            utilities.log_this(experiments_set.logs_path, f"fold {curr_fold}/{num_folds}", p=False)
            
            # set the current fold into the experiment class
            experiment.set_current_fold(curr_fold, train_folds, test_fold)

            # extract groups and labels for the train fold
            train_groups = [exp_ds.groups[movie] for movie in train_folds]
            train_labels = [exp_ds.labels[movie] for movie in train_folds]

            # extract training and test patients from the movies (and saves the split in a JSON file)
            train_pats, test_pats = experiment.from_fold_split_to_pats(train_folds, test_fold)
            
            # extract folds labels
            _, fold_train_y = exp_ds.build_tfrecord_from_patients(list(train_pats))
            _, fold_test_y = exp_ds.build_tfrecord_from_patients(list(test_pats))

            # # TODO: da rimuovere
            # train_class_weight = compute_class_weight('balanced', classes=np.unique(fold_train_y), y=fold_train_y)
            # test_class_weight = compute_class_weight('balanced', classes=np.unique(fold_test_y), y=fold_test_y)
            
            # class_weights_collection.append(train_class_weight)
            # class_weights_collection.append(test_class_weight)
            
            # print(f"train_class_weight: {train_class_weight}")
            # print(f"test_class_weight: {test_class_weight}")

            # print split charts of the current fold
            utilities.plot_pats_fold(experiment, train_pats, test_pats)
            utilities.plot_fdistr_per_class(experiment, fold_train_y, fold_test_y, phase='fold')
            
            # sample a label for each patient on the train folds based on the movie-level ranking
            labels_pats = exp_ds.sample_score_per_patient(train_groups, train_labels)

            # ~~~~~~~~~~~~~~~~~~~~~~~~ GRID SEARCH HOLDOUTS ~~~~~~~~~~~~~~~~~~~~~~~~~
            if not args.no_gridsearch:
                # perform hyperparameters grid search if enabled
                hpv_sss = exp_ds.n_strat_shuffle_split(train_pats, labels_pats, val_ratio=0.15, splits=args.hpv_splits, state=f"{curr_fold}_hpv")
                for hpv_curr_split, (hpv_train, hpv_val) in enumerate(hpv_sss, start=1):
                    print(f"******** Grid Search Holdout {hpv_curr_split}/{args.hpv_splits} *******")
                    utilities.log_this(experiments_set.logs_path, f"grid search holdout {hpv_curr_split}/{args.hpv_splits}", p=False)

                    # extract patients from the splitting
                    hpv_train = [train_pats[pat] for pat in hpv_train]
                    hpv_val = [train_pats[pat] for pat in hpv_val]

                    # set the current holdout for hyperparameters grid searching  
                    experiment.set_current_hpv_holdout(hpv_curr_split, hpv_train, hpv_val)

                    # prepare train and val sets from the patients got from the holdout
                    experiment.prepare_hpv_sets()

                    # compute the grid of every parameters combination
                    hpv_grid = experiment.get_hyperparameters_grid()
                    
                    print(hpv_grid)

                    # gridsearching...
                    for hp_iter, hyperparameters in enumerate(hpv_grid, start=1):
                        print(f"testing HPs [{hp_iter}/{len(hpv_grid)}]")

                        experiment.evaluate_hyperparams(hyperparameters, epochs=5)

                    # save the MAE values of each parameter combination for the current holdout
                    experiment.splits_mae_scores[str(hpv_curr_split)] = experiment.mae_scores
                
                # Find the best combination of hyper-parameters from the grid search
                best_params = experiment.compute_grid_search_results()
            else:
                # load default good parameters combination for each network model
                # Define default parameters for each network model
                # default_params = {
                #     'obd': {'batch_size': 32, 'dropout': 0.3, 'learning_rate': 'cdr', 'hidden_size': 512},
                #     'clm': {'batch_size': 32, 'dropout': 0.0, 'learning_rate': 0.01, 'link_function': 'logit', 'use_tau': True},
                #     'default': {'batch_size': 32, 'dropout': 0.0, 'learning_rate': 0.01}
                # }

                # Load default parameters based on the experiment's network model
                # best_params = exp_params.get(experiment.settings['nn_model'], exp_params['default'])

                # Load default parameters based on the experiment's network model
                exp_sett = experiment.settings
                if experiment.settings['nn_model'] == 'obd':
                    best_params = {'batch_size': exp_sett["batch_size"][0], 'dropout': exp_sett["dropout"][0], 'learning_rate': exp_sett["learning_rate"][0], 'hidden_size': exp_sett["hidden_size"][0]}
                elif experiment.settings['nn_model'] == 'clm':
                    best_params = {'batch_size': exp_sett["batch_size"][0], 'dropout': exp_sett["dropout"][0], 'learning_rate': exp_sett["learning_rate"][0], 'link_function': exp_sett["link_function"][0], 'use_tau': exp_sett["use_tau"][0]}
                else:
                    best_params = {'batch_size': exp_sett["batch_size"][0], 'dropout': exp_sett["dropout"][0], 'learning_rate': exp_sett["learning_rate"][0]}

            print(best_params)

            # ~~~~~~~~~~~~~~~~~~~~~~ NETWORK TRAINING HOLDOUTS ~~~~~~~~~~~~~~~~~~~~~~
            hpt_sss = exp_ds.n_strat_shuffle_split(train_pats, labels_pats, val_ratio=0.15, splits=args.hpt_splits, state=f"{curr_fold}_hpt")
            for hpt_curr_split, (hpt_train, hpt_val) in enumerate(hpt_sss, start=1):
                print(f"****** Model Training Holdout {hpt_curr_split}/{args.hpt_splits} ******")
                utilities.log_this(experiments_set.logs_path, f"training holdout {hpt_curr_split}/{args.hpt_splits}", p=False)

                # extract patients from the splitting
                hpt_train = [train_pats[pat] for pat in hpt_train]
                hpt_val = [train_pats[pat] for pat in hpt_val]

                # set the current training holdout in the experiment
                experiment.set_current_hpt_holdout(hpt_curr_split, hpt_train, hpt_val)

                # prepare train and val sets from the patients got from the holdout
                experiment.prepare_hpt_sets()
                
                # # TODO: da rimuovere
                # hpt_train_class_weight = compute_class_weight('balanced', classes=np.unique(experiment.y_hpt_train), y=experiment.y_hpt_train)
                # hpt_test_class_weight = compute_class_weight('balanced', classes=np.unique(experiment.y_hpt_val), y=experiment.y_hpt_val)
                
                # class_weights_collection.append(hpt_train_class_weight)
                # class_weights_collection.append(hpt_test_class_weight)

                # print(f"hpt_train_class_weight: {hpt_train_class_weight}")
                # print(f"hpt_test_class_weight: {hpt_test_class_weight}")

                # print the patients splitting in the train and test sets for the current fold
                utilities.plot_fdistr_per_class(experiment, phase='hpt')
                
                #hpt_train = exp_ds.generate_tfrset(experiment.hpt_train, batch_size=experiment.settings['batch_size'][0], shuffle=True, augment=True)
                #exp_ds.plot_set_batches(hpt_train, experiment.settings['batch_size'][0])
                
                # continue

                # train the network with the best parameters on the holdout and get the history 
                hpt_model, hpt_history = experiment.hpt_train_network(best_params)

                # save the training graphs
                experiment.nn_train_graphs(hpt_history)

                # test the model on the testing fold (from the initial k-fold)
                experiment.hpt_test_network(test_pats, hpt_model, best_params)

        print("\n")

        # # TODO: da rimuovere
        # reference_class_weights = np.array([0.73754569, 1.03050017, 0.77646005, 2.5916608])
        # class_weights_with_different_shape, mean_class_weights, std_class_weights = utilities.compute_class_weights_mean(class_weights_collection)
        # distance = utilities.euclidean_distance(mean_class_weights, reference_class_weights)

        # with open("class_weights_seed.txt", "a") as file:
        #     file.write(f"Seed: {args.seed}\n")
        #     file.write(f"Class Weights with different shape ({len(class_weights_with_different_shape)}): {class_weights_with_different_shape}\n")
        #     file.write(f"Reference Class Weights:\t{reference_class_weights}\n")
        #     file.write(f"Mean Class Weights:\t\t{mean_class_weights}\n")
        #     file.write(f"St.Dev. Class Weights:\t\t{std_class_weights}\n")
        #     file.write(f"Euclidean Distance:\t\t{distance}\n\n")

        # break

    utilities.log_this(experiments_set.logs_path, f"End", p=False)
    print("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ END ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★\n")
    
if __name__ == "__main__":
    main()