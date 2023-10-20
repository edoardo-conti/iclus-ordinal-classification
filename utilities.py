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


def plot_split_graphs(train_subset, val_subset, test_subset, ds_info):
    # Plot the distribution of frames per center
    frame_counts = ds_info['frames_by_center']
    centers = list(frame_counts.keys())
    frame_values = [frame_counts[center] for center in centers]

    plt.figure(figsize=(12, 6))
    plt.bar(centers, frame_values)
    plt.title('Distribution of Frames per Medical Center')
    plt.xlabel('Medical Center')
    plt.ylabel('Number of Frames')
    plt.yscale('log')
    plt.xticks(rotation=45, fontsize=8)
    plt.show()

    # Plot the distribution of patients per center
    train_patients = ds_info['train_patients_by_center']
    val_patients = ds_info['val_patients_by_center']
    test_patients = ds_info['test_patients_by_center']

    train_values = [len(train_patients[center]) for center in centers]
    val_values = [len(val_patients[center]) for center in centers]
    test_values = [len(test_patients[center]) for center in centers]

    width = 0.3
    x = range(len(centers))

    plt.figure(figsize=(12, 6))
    plt.bar(x, train_values, width, label='Train set', align='center')
    plt.bar([i + width for i in x], val_values, width, label='Val set', align='center')
    plt.bar([i + 2 * width for i in x], test_values, width, label='Test set', align='center')
    plt.title('Distribution of Patients per Medical Center')
    plt.xlabel('Medical Center')
    plt.ylabel('Number of Patients')
    plt.yscale('log')
    plt.xticks([i + width for i in x], centers, rotation=45, fontsize=8)
    plt.legend()
    plt.show()    