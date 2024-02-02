import os
import numpy as np
import tensorflow as tf
import keras 
import matplotlib
import matplotlib.pyplot as plt

class GradCAMCallback(tf.keras.callbacks.Callback):
    def __init__(self, 
                 model, 
                 experiment,
                 val_data = None,
                 nsamples_per_class = 1, 
                 freq = 5):
        super(GradCAMCallback, self).__init__()
        self.model = model
        self.val_data = val_data
        self.num_classes = experiment.ds_num_classes
        self.conv_layer_name = experiment.last_conv_layer
        self.show_cams = experiment.output_mode[0]
        self.save_cams = experiment.output_mode[1]
        self.freq = freq
        
        # gather 'x' samples for each class
        self.samples_per_class = self.get_classes_samples(nsamples_per_class)
        
        # compute gradcams save path
        self.gradcams_save_path = os.path.join(experiment.hpt_holdout_dir, "gradcams/")
    

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq != 0:
            return

        images_with_gradcam = []

        for label, sample in self.samples_per_class.items():
            image_array = self.get_img_array(sample[0])
            gradcam = self.make_gradcam_heatmap(image_array, self.model, self.conv_layer_name, pred_index=label)
            gradcam_img_merged = self.merge_gradcam(sample[0], gradcam)
            images_with_gradcam.append(gradcam_img_merged)

        _, axs = plt.subplots(1, 4, figsize=(15, 80))
        plt.subplots_adjust(wspace=0.5)

        for i, ax in enumerate(axs.ravel()):
            ax.imshow(images_with_gradcam[i], cmap='gray')
            ax.set_title(f"class {i}")

        if self.show_cams:
            plt.show()

        if self.save_cams:
            if not os.path.exists(self.gradcams_save_path):
                os.makedirs(self.gradcams_save_path)
            current_gradcam_path = os.path.join(self.gradcams_save_path, f"gradcam_epoch_{epoch+1}.png")
            plt.savefig(current_gradcam_path, bbox_inches='tight', pad_inches=0.2)
        
        plt.close()


    def get_classes_samples(self, nsamples_per_class=1):
        samples_per_class = {0: [], 1: [], 2: [], 3: []}

        for images, labels in self.val_data:
            for image, label in zip(images, labels):
                class_index = tf.argmax(label).numpy()
                if len(samples_per_class[class_index]) < nsamples_per_class:
                    samples_per_class[class_index].append(image)

            if all(len(samples) >= nsamples_per_class for samples in samples_per_class.values()):
                break

        return samples_per_class


    def get_img_array(self, img):
        return np.expand_dims(keras.utils.img_to_array(img), axis=0)


    def make_gradcam_heatmap(self, img_array, model, last_conv_layer_name, pred_index=None):
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = keras.models.Model(model.inputs, [last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            pred_index = pred_index or tf.argmax(preds[0])
            pred_index -= tf.cast(self.model.name == 'obd' and pred_index == 3, tf.int64)

            class_channel = preds[:, pred_index]
        
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

        return heatmap.numpy()


    def merge_gradcam(self, img, heatmap, alpha=0.4):
        img = keras.utils.img_to_array(img)
        heatmap = np.uint8(255 * np.nan_to_num(heatmap, nan=0))

        jet = matplotlib.colormaps.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)
        
        jet_heatmap = jet_heatmap / 255
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.utils.array_to_img(superimposed_img)

        return superimposed_img