import numpy as np
import tensorflow as tf
import keras 
from sklearn.preprocessing import normalize
import matplotlib
import matplotlib.pyplot as plt

class GradCAMCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, layer_name, val_data, nsamples_per_class=1, freq=5):
        super(GradCAMCallback, self).__init__()
        self.model = model
        self.layer_name = layer_name
        self.val_data = val_data
        self.freq = freq

        # gather 'x' samples for each class
        self.samples_per_class = self.get_classes_samples(nsamples_per_class)
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq == 0:
            images_with_gradcam = []

            for _, sample in self.samples_per_class.items():
                image_array = self.get_img_array(sample[0])
                gradcam = self.make_gradcam_heatmap(image_array, self.model, self.layer_name)
                gradcam_img_merged = self.merge_gradcam(sample[0], gradcam)
                images_with_gradcam.append(gradcam_img_merged)

            _, axs = plt.subplots(1, 4, figsize=(15, 60))
            plt.subplots_adjust(wspace=0.4)

            for i, ax in enumerate(axs.ravel()):
                ax.imshow(images_with_gradcam[i], cmap='gray')
                ax.set_title(f"class {i}")
                #ax.axis('off')

            plt.show()

    def get_classes_samples(self, nsamples_per_class=1):
        # Dizionario per tenere traccia dei campioni estratti per ciascuna classe
        samples_per_class = {0: [], 1: [], 2: [], 3: []}

        # Indice del batch da esaminare
        batch_index = 0

        while True:
            try:
                images, labels = self.val_data[batch_index]
                for image, label in zip(images, labels):
                    class_index = tf.argmax(label).numpy()
                    if len(samples_per_class[class_index]) < nsamples_per_class:
                        samples_per_class[class_index].append(image)
                
                if all(len(samples) >= nsamples_per_class for samples in samples_per_class.values()):
                    break
                
                batch_index += 1
            except StopIteration:
                break
    
        return samples_per_class

    def get_img_array(self, img):
        array = keras.utils.img_to_array(img)
        array = np.expand_dims(array, axis=0)

        return array

    def make_gradcam_heatmap(self, img_array, model, last_conv_layer_name, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = keras.models.Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        return heatmap.numpy()

    def merge_gradcam(self, img, heatmap, alpha=0.4):
        # Load the original image
        #img = keras.utils.load_img(img_path)
        img = keras.utils.img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = matplotlib.colormaps.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        jet_heatmap = jet_heatmap / 255
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.utils.array_to_img(superimposed_img)

        return superimposed_img
