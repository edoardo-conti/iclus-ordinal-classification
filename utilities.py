import os
import datetime
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------> /START WIP <-------------------------------

def compute_class_weights_mean(class_weights_list):
    # Inizializza una lista per registrare il conteggio dei class weights con shape diversa
    class_weights_with_different_shape = []

    # Crea una nuova lista contenente solo gli elementi che soddisfano la condizione
    valid_class_weights = [np.array(class_weights) for class_weights in class_weights_list if len(class_weights) == 4]

    # Controlla se ci sono class weights con shape diversa
    for class_weights in class_weights_list:
        if np.array(class_weights).shape != (4,):
            class_weights_with_different_shape.append(np.array(class_weights))

    # Calcola la media lungo l'asse 0 (media per ogni classe)
    class_weights_mean = np.mean(valid_class_weights, axis=0)
    class_weights_std = np.std(valid_class_weights, axis=0)
    
    return class_weights_with_different_shape, class_weights_mean, class_weights_std


def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum((vector1 - vector2)**2))


def plot_pats_fold(exp, train_pats, test_pats):
    display, save = exp.output_mode
    fold_split = {'train':train_pats, 'test':test_pats}
 
    centri_medici = list(set([paziente.split('/')[0] for split, pazienti in fold_split.items() for paziente in pazienti]))
    sets = list(fold_split.keys())
    
    # Conta il numero di pazienti per centro medico e set
    counts = np.zeros((len(centri_medici), len(sets)))
    for i, centro_medico in enumerate(centri_medici):
        for j, split in enumerate(sets):
            counts[i, j] = sum(1 for paziente in fold_split[split] if paziente.startswith(centro_medico))

    # Crea il grafico a barre impilato
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(centri_medici))
    for j, split in enumerate(sets):
        plt.bar(centri_medici, counts[:, j], bottom=bottom, label=split)
        bottom += counts[:, j]

    plt.title('Patient Distribution by Medical Center in sets')
    plt.xlabel('Medical center')
    plt.ylabel('Patient count')
    plt.legend()
    plt.xticks(rotation=45, ha='right')

    if save:
        chart_file_path = os.path.join(exp.fold_subdir, "split_per_patients.png")
        plt.savefig(chart_file_path, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    # Display the plot
    if display:
        plt.show()


def plot_fdistr_per_class(exp, y_seta=None, y_setb=None, phase='fold'):
    display, save = exp.output_mode

    if phase == 'fold':
        sets = ['Train', 'Test']
        datasets = [y_seta, y_setb]
        chart_file_path = os.path.join(exp.fold_subdir, "frames_distr_per_class_pie.png")
    elif phase == 'hpt':
        sets = ['Train', 'Validation']
        datasets = [exp.y_hpt_train, exp.y_hpt_val]
        chart_file_path = os.path.join(exp.hpt_holdout_dir, "frames_distr_per_class_pie.png")
    else:
        sets = ['Train', 'Validation']
        datasets = [exp.y_hpv_train, exp.y_hpv_val]
        chart_file_path = os.path.join(exp.hpv_holdout_dir, "frames_distr_per_class_pie.png")

    tot_labels = len(datasets[0]) + len(datasets[1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for i, set_name in enumerate(sets):
        class_counts = np.bincount(datasets[i])
        labels = np.arange(len(class_counts)).astype(int)

        wedges, _, _ = axes[i].pie(class_counts, labels=labels, autopct=lambda p: '{:.1f}%\n({:.0f})'.format(p, p * sum(class_counts) / 100), startangle=90)
        axes[i].set_title(f'{set_name} Set ({round(sum(class_counts) * 100 / tot_labels)}%)')

    # Creazione di una legenda unica per tutta la figura
    legend_labels = [f'Class {label}' for label in labels]
    fig.legend(wedges, legend_labels, title='Classes', loc='lower center', ncol=len(set_name))
    
    plt.suptitle('Frames distribution in sets for each class', y=1.05)
    
    if save:
        plt.savefig(chart_file_path, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    # display the plot
    if display:
        plt.show()


def log_this(logs_path, message, p=True):
    # get the current date and time
    timelog = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # print the log message
    if p:
        print(message)
    
    # log the start of the run
    with open(logs_path, 'a') as log_file: log_file.write(f"{timelog} - {message}\n")


# -------------------------------> /END WIP <--------------------------------


def plot_labels_per_patient_hists(labels_per_patient, display=False):
    num_patients = len(labels_per_patient)
    rows = 5
    cols = 7

    _, axs = plt.subplots(rows, cols, figsize=(15, 10))

    for i in range(rows):
        for j in range(cols):
            patient_idx = i * cols + j
            if patient_idx < num_patients:
                patient_key = list(labels_per_patient.keys())[patient_idx]
                labels = labels_per_patient[patient_key]
    
                axs[i, j].hist(labels, bins=np.arange(5) - 0.5, edgecolor='black', linewidth=1.2)
                axs[i, j].set_title(patient_key)
                axs[i, j].set_xticks(range(5))
                axs[i, j].set_xticklabels([str(k) for k in range(5)])
                axs[i, j].set_xlabel('Score')
                axs[i, j].set_ylabel('Frequency')

    plt.tight_layout()
    plt.title('Labels per patients distribution')

    # Display the plot
    if display:
        plt.show()

    return plt.gcf()

def plot_fdistr_per_class_pie(y_train_ds, y_val_ds, y_test_ds, display=False):
    sets = ['Train', 'Validation', 'Test']
    datasets = [y_train_ds, y_val_ds, y_test_ds]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, set_name in enumerate(sets):
        class_counts = np.bincount(datasets[i])
        labels = np.arange(len(class_counts)).astype(int)

        wedges, _, autotexts = axes[i].pie(class_counts, labels=labels, autopct=lambda p: '{:.1f}%\n({:.0f})'.format(p, p * sum(class_counts) / 100), startangle=90)
        axes[i].set_title(f'{set_name} Set')

        # Aggiungi etichette con il numero di frame per ogni fetta
        #for autotext in autotexts:
            #autotext.set_color('white')  # Imposta il colore del testo a bianco per una migliore leggibilità

    # Creazione di una legenda unica per tutta la figura
    legend_labels = [f'Class {label}' for label in labels]
    fig.legend(wedges, legend_labels, title='Classes', loc='lower center', ncol=len(set_name))

    plt.suptitle('Frames distribution in sets for each class', y=1.05)

    # display the plot
    if display:
        plt.show()

    return plt.gcf()


def plot_labels_distr(labels, display=False):
    # create an occurrence count of each class
    counts = {label: labels.count(label) for label in set(labels)}

    # converts the count into two separate lists for plotting
    class_names, class_counts = zip(*counts.items())

    # create a bar-plot 
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_counts)
    plt.title('Labels distribution in the dataset')
    plt.xlabel('Classes')
    plt.ylabel('Frames')
    
    # display the plot
    if display:
        plt.show()
    
    return plt.gcf()


def plot_charts(exp, charts, display, save, save_path):
    # create the charts subfolder 
    charts_path = os.path.join(save_path, 'charts/') 
    os.makedirs(charts_path, exist_ok=True)

    #if "splitinfo" in charts:
        #print_split_ds_info(exp.dataset_metadata)
    
    if "pdistr" in charts:
        pps = plot_patients_split(exp.dataset.split, display=display)
        if save:
            chart_file_path = os.path.join(charts_path, "split_per_patients.png")
            pps.savefig(chart_file_path)
            plt.close()
    
    if "lsdistr_pie" in charts:
        pfpcp = plot_fdistr_per_class_pie(exp.y_train, exp.y_val, exp.y_test, display=display)
        if save:
            chart_file_path = os.path.join(charts_path, "frames_distr_per_class_pie.png")
            pfpcp.savefig(chart_file_path)
            plt.close()

    if "ldistr" in charts:
        ds_labels = list(exp.y_train) + list(exp.y_val) + list(exp.y_test)
        pld = plot_labels_distr(ds_labels, display=display)
        if save:
            chart_file_path = os.path.join(charts_path, "labels_distr.png")
            pld.savefig(chart_file_path)
            plt.close()