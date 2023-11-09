import numpy as np
import matplotlib.pyplot as plt

def print_split_ds_info(ds_info):
    # Print textual split information
    for medical_center in ds_info['medical_center_patients'].keys():
        print(f"Medical Center: {medical_center}")
        print(f"  Frames in center: {ds_info['frames_by_center'][medical_center]}")
        print(f"  Train patients:")
        for patient in ds_info['train_patients_by_center'][medical_center]:
            frame_count = ds_info['frames_by_center_patient'][medical_center][patient]
            print(f"   {patient}: {frame_count} frames")
        print(f"  Val patients:")
        for patient in ds_info['val_patients_by_center'][medical_center]:
            frame_count = ds_info['frames_by_center_patient'][medical_center][patient]
            print(f"   {patient}: {frame_count} frames")
        print(f"  Test patients:")
        for patient in ds_info['test_patients_by_center'][medical_center]:
            frame_count = ds_info['frames_by_center_patient'][medical_center][patient]
            print(f"   {patient}: {frame_count} frames")


def plot_frames_split(ds_info, log_scale=False, display=False):
    # Create data for the plot
    centers = []
    train_frame_counts = []
    val_frame_counts = []
    test_frame_counts = []

    for medical_center in ds_info['medical_center_patients'].keys():
        centers.append(medical_center)
        train_frame_count = sum(ds_info['frames_by_center_patient'][medical_center][patient] for patient in ds_info['train_patients_by_center'][medical_center])
        val_frame_count = sum(ds_info['frames_by_center_patient'][medical_center][patient] for patient in ds_info['val_patients_by_center'][medical_center])
        test_frame_count = sum(ds_info['frames_by_center_patient'][medical_center][patient] for patient in ds_info['test_patients_by_center'][medical_center])
        train_frame_counts.append(train_frame_count)
        val_frame_counts.append(val_frame_count)
        test_frame_counts.append(test_frame_count)

    # Create the plot
    plt.figure(figsize=(10, 6))
    if log_scale:
        plt.barh(centers, train_frame_counts, label='Train frames', log=True)
        plt.barh(centers, val_frame_counts, left=train_frame_counts, label='Val frames', log=True)
        plt.barh(centers, test_frame_counts, left=[sum(x) for x in zip(train_frame_counts, val_frame_counts)], label='Test frames', log=True)
    else:
        plt.barh(centers, train_frame_counts, label='Train frames')
        plt.barh(centers, val_frame_counts, left=train_frame_counts, label='Val frames')
        plt.barh(centers, test_frame_counts, left=[sum(x) for x in zip(train_frame_counts, val_frame_counts)], label='Test frames')

    # Add labels and legend
    plt.xlabel('Frame Count (Log Scale)' if log_scale else 'Frame Count')
    plt.ylabel('Medical Center')
    plt.title('Frame Distribution by Medical Center')
    plt.legend()

    # display the plot
    if display:
        plt.show()

    return plt.gcf()


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

def plot_labels_distr(y_train_ds, y_val_ds, y_test_ds, display=False):
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
    plt.title('Distribution of labels for each set')
    plt.xticks(group_labels, [0, 1, 2, 3])
    plt.legend()  

    # display the plot
    if display:
        plt.show()
    
    return plt.gcf()
 