import os
import argparse
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import backend as K
from tqdm import tqdm
from network.model import NeuralNetwork

from network.losses import  make_cost_matrix, qwk_loss

# Imposta il seed per garantire la riproducibilità
def set_random_seed(seed_value=42):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def count_videos(dataset_dir):
    """
    Conta il numero totale di video (.mp4) nelle sottocartelle dell'ospedale, ignorando file non rilevanti.
    """
    total_videos = 0
    for hospital_name in os.listdir(dataset_dir):
        hospital_path = os.path.join(dataset_dir, hospital_name)
        cropped_dir = os.path.join(hospital_path, f"{hospital_name}_Cropped")
        
        if not os.path.isdir(cropped_dir):
            continue  # Salta se non è una directory
        
        for patient_id in os.listdir(cropped_dir):
            patient_path = os.path.join(cropped_dir, patient_id)
            if not os.path.isdir(patient_path):
                continue  # Salta se non è una directory
            
            for analysis_id in os.listdir(patient_path):
                analysis_path = os.path.join(patient_path, analysis_id)
                if not os.path.isdir(analysis_path):
                    continue  # Salta se non è una directory
                
                for video_file in os.listdir(analysis_path):
                    if video_file.endswith('.mp4'):
                        total_videos += 1
    
    return total_videos


def extract_frames_from_video(video_path):
    """
    Estrae i frame da un video .mp4 e li restituisce come array di immagini (3 canali).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess con OpenCV
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame_resized = cv2.resize(frame_rgb, (224, 224)) / 255.0
        
        # preprocess
        frame_resized = tf.image.resize(frame, [224, 224])  / 255.0

        frames.append(frame_resized)

    cap.release()

    # return tf.stack(frames)
    return np.array(frames)


def save_predictions_to_excel(predictions, hospital_name, patient_id, analysis_id, area_code, output_dir):
    """
    Salva le predizioni frame-by-frame in un file Excel con due colonne:
    - Colonna 1: Numero del frame
    - Colonna 2: Label predetta
    """
    predicted_labels = np.argmax(predictions, axis=1)
    frame_numbers = np.arange(0, len(predicted_labels))

    data = {'Frame': frame_numbers, 'Label': predicted_labels}

    # Creare il nome del file basato su patient_id e area_code
    filename = f"{patient_id}_{area_code}.xlsx"

    # Creare la directory gerarchica basata su ospedale/paziente/analisi
    output_path = os.path.join(output_dir, hospital_name, str(patient_id), str(analysis_id))
    os.makedirs(output_path, exist_ok=True)

    # Salva il file Excel nella directory corretta
    file_path = os.path.join(output_path, filename)
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)


def get_area_code_from_filename(filename):
    """
    Estrae l'Area Code dal nome del file video.
    Il formato del nome è 'clipped_AnalysisID_AreaCode.mp4'.
    Restituisce l'ultimo valore numerico (Area Code) tra l'ultimo underscore e '.mp4'.
    """
    # Rimuovi l'estensione del file
    base_name = os.path.splitext(filename)[0]
    
    # Dividi il nome del file usando l'underscore come separatore
    parts = base_name.split('_')
    
    # L'Area Code è l'ultimo elemento della lista
    area_code = parts[-1]
    
    return area_code


def evaluate_on_dataset(dataset_dir, output_dir, model, selected_hospitals=None, one_shot=False):
    """
    Naviga la struttura delle cartelle per processare i video e generare predizioni.
    """
    total_videos = count_videos(dataset_dir)

    with tqdm(total=total_videos, desc="Processing Videos") as pbar:
        for hospital_name in sorted(os.listdir(dataset_dir)):
            if selected_hospitals and hospital_name not in selected_hospitals:
                continue  # Salta se l'ospedale non è nella lista selezionata

            hospital_path = os.path.join(dataset_dir, hospital_name)
            if not os.path.isdir(hospital_path):
                continue
            
            # Cerca la sottocartella 'Hospital_Cropped'
            cropped_dir = os.path.join(hospital_path, f"{hospital_name}_Cropped")
            if not os.path.isdir(cropped_dir):
                print(f"Errore: La cartella '{hospital_name}_Cropped' non esiste in '{hospital_path}'")
                continue
            
            # Per ogni cartella paziente in 'Hospital_Cropped'
            for patient_id in sorted(os.listdir(cropped_dir)):
                patient_path = os.path.join(cropped_dir, patient_id)
                if not os.path.isdir(patient_path):
                    continue
                
                # Per ogni cartella di analisi
                for analysis_id in sorted(os.listdir(patient_path)):
                    analysis_path = os.path.join(patient_path, analysis_id)
                    if not os.path.isdir(analysis_path):
                        continue

                    # Processa ogni video nell'analisi
                    for video_file in sorted(os.listdir(analysis_path)):
                        if video_file.endswith('.mp4'):
                            area_code = get_area_code_from_filename(video_file)
                            video_path = os.path.join(analysis_path, video_file)

                            print(f"Processing: Hospital={hospital_name}, Patient={patient_id}, Analysis={analysis_id}, Area={area_code}")

                            # Estrai i frame dal video
                            frames = extract_frames_from_video(video_path)

                            # Verifica se ci sono frame
                            if len(frames) == 0:
                                print(f"Errore: Nessun frame trovato nel video {video_file}")
                                continue
                            
                            # Esegui la predizione sui frame
                            # predictions = model.predict(frames)
                            predictions = model(frames, training=False)

                            # Converti le predizioni in etichette
                            # predicted_labels = np.argmax(predictions, axis=1)

                            # Salva le predizioni in Excel
                            save_predictions_to_excel(predictions, hospital_name, patient_id, analysis_id, area_code, output_dir)
                            
                            # Aggiorna la barra di progresso
                            pbar.update(1)

                            if one_shot:
                                return

def main():
    set_random_seed(42)

    parser = argparse.ArgumentParser(description="LUS Ordinal Classification - Evaluation")

    # Parametro per il path dei pesi migliori
    parser.add_argument('--weights_path', type=str, required=True, help='Path al file dei migliori pesi salvati (.h5)')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory del dataset di evaluation')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory di output')

    args = parser.parse_args()

    ################################# MODELLO NEURALE ######################################## 
    # Parametri del modello
    MODEL = 'clm' 

    # Istanza della rete neurale
    network = NeuralNetwork()
    model = network.build(MODEL)
    
    # cost_matrix = K.constant(make_cost_matrix(4), dtype=K.floatx())
    # loss = qwk_loss(cost_matrix)
    # model.compile(optimizer="SGD", loss=loss)

    # Carica i pesi salvati dal path fornito
    model.load_weights(args.weights_path)
    
    #################################### EVALUATION ########################################## 
    
    # Valutazione su dataset esterno con salvataggio delle predizioni
    evaluate_on_dataset(args.dataset_dir, args.output_dir, model, one_shot=False)

if __name__ == "__main__":
    main()