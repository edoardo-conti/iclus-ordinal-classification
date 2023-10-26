import numpy as np
import matplotlib.pyplot as plt

def print_split_diagnostic_info(ds_info):
    # Print diagnostic information
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


def plot_fsplit_info(ds_info, log_scale=False):
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
        plt.barh(centers, train_frame_counts, label='Train Frames', log=True)
        plt.barh(centers, val_frame_counts, left=train_frame_counts, label='Val Frames', log=True)
        plt.barh(centers, test_frame_counts, left=[sum(x) for x in zip(train_frame_counts, val_frame_counts)], label='Test Patients', log=True)
    else:
        plt.barh(centers, train_frame_counts, label='Train Frames')
        plt.barh(centers, val_frame_counts, left=train_frame_counts, label='Val Frames')
        plt.barh(centers, test_frame_counts, left=[sum(x) for x in zip(train_frame_counts, val_frame_counts)], label='Test Patients')

    # Add labels and legend
    plt.xlabel('Frame Count (Log Scale)' if log_scale else 'Frame Count')
    plt.ylabel('Medical Center')
    plt.title('Frame Distribution by Medical Center')
    plt.legend()

    # Show the plot
    plt.show()


def plot_psplit_info(ds_info):
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
    plt.barh(centers, train_patient_counts, label='Train Patients')
    plt.barh(centers, val_patient_counts, left=train_patient_counts, label='Val Patients')
    plt.barh(centers, test_patient_counts, left=[sum(x) for x in zip(train_patient_counts, val_patient_counts)], label='Test Patients')

    # Add labels and legend
    plt.xlabel('Patient Count')
    plt.ylabel('Medical Center')
    plt.title('Patient Distribution by Medical Center')
    plt.legend()

    # Show the plot
    plt.show()
