import numpy as np
import matplotlib.pyplot as plt

def print_split_ds_info(ds_info):
    # Print textual split information
    for medical_center in ds_info['medical_center_patients'].keys():
        print(f"Medical Center: {medical_center}")
        print(f"  Frames in center: {ds_info['frames_by_center'][medical_center]}")
        print("  Train patients:")
        for patient in ds_info['train_patients_by_center'][medical_center]:
            frame_count = ds_info['frames_by_center_patient'][medical_center][patient]
            print(f"   {patient}: {frame_count} frames")
        print("  Val patients:")
        for patient in ds_info['val_patients_by_center'][medical_center]:
            frame_count = ds_info['frames_by_center_patient'][medical_center][patient]
            print(f"   {patient}: {frame_count} frames")
        print("  Test patients:")
        for patient in ds_info['test_patients_by_center'][medical_center]:
            frame_count = ds_info['frames_by_center_patient'][medical_center][patient]
            print(f"   {patient}: {frame_count} frames")


def plot_patients_split(ds_info, display=False):
    # Create data for the plot
    centers = []
    train_patient_counts = []
    val_patient_counts = []
    test_patient_counts = []

    for medical_center in ds_info['medical_center_patients'].keys():
        centers.append(medical_center)
        train_patient_count = len(ds_info['train_patients_by_center'][medical_center])
        val_patient_count = len(ds_info['val_patients_by_center'][medical_center])
        test_patient_count = len(ds_info['test_patients_by_center'][medical_center])
        train_patient_counts.append(train_patient_count)
        val_patient_counts.append(val_patient_count)
        test_patient_counts.append(test_patient_count)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.barh(centers, train_patient_counts, label='Train patients')
    plt.barh(centers, val_patient_counts, left=train_patient_counts, label='Val patients')
    plt.barh(centers, test_patient_counts, left=[sum(x) for x in zip(train_patient_counts, val_patient_counts)], label='Test patients')

    # Add labels, title and legend
    plt.xlabel('Patient Count')
    plt.ylabel('Medical Center')
    plt.title('Patient Distribution by Medical Center')
    plt.legend()

    # display the plot
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

        wedges, _, _ = axes[i].pie(class_counts, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[i].set_title(f'{set_name} Set')

     # Creazione di una legenda unica per tutta la figura
    legend_labels = [f'Class {label}' for label in labels]
    fig.legend(wedges, legend_labels, title='Classes', loc='lower center', ncol=len(set_name))

    plt.suptitle('Frames distribution in sets for each class', y=1.05)
    #plt.subplots_adjust(bottom=0.3)
    
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