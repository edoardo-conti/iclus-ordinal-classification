import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

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


def plot_patients_split(dataset_split, display=False):
    # Estrai i centri medici e i pazienti dai dati
    centri_medici = list(set([paziente.split('/')[0] for split, pazienti in dataset_split.items() for paziente in pazienti]))
    sets = list(dataset_split.keys())
    
    # Conta il numero di pazienti per centro medico e set
    counts = np.zeros((len(centri_medici), len(sets)))
    for i, centro_medico in enumerate(centri_medici):
        for j, split in enumerate(sets):
            counts[i, j] = sum(1 for paziente in dataset_split[split] if paziente.startswith(centro_medico))

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

    # Display the plot
    if display:
        plt.show()

    return plt.gcf()


def plot_fdistr_per_class(y_train_ds, y_val_ds, y_test_ds, display=False):
    # calculate the class count for each set
    class_counts_val = np.bincount(y_val_ds)
    class_counts_test = np.bincount(y_test_ds)
    class_counts_train = np.bincount(y_train_ds)

    class_labels = np.arange(len(class_counts_val)).astype(int)
    group_labels = np.arange(len(class_labels))
    bar_width = 0.2

    bar_positions_train = class_labels - bar_width
    bar_positions_val = class_labels
    bar_positions_test = class_labels + bar_width

    # Create the plot
    plt.bar(bar_positions_train, class_counts_train, width=bar_width, label='Train frames')
    plt.bar(bar_positions_val, class_counts_val, width=bar_width, label='Validation frames')
    plt.bar(bar_positions_test, class_counts_test, width=bar_width, label='Test frames')
    
    # Add labels, title and legend
    plt.xlabel('Classes')
    plt.ylabel('Frames')
    plt.title('Frames distribution in sets for each class')
    plt.xticks(group_labels, [0, 1, 2, 3])
    plt.legend()  

    # display the plot
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