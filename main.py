import kivy
from kivy.app import App
import tensorflow as tf
from keras import models
import numpy as np

class_to_number = {0 : 1, 1 : 10, 2 : 11, 3 : 12, 4 : 13, 5 : 14, 6 : 15, 7 : 16, 8 : 17, 9 : 18, 10 : 19,
	                 11 : 2, 12 : 20, 13 : 21, 14 : 22, 15 : 23, 16 : 24, 17 : 25, 18 : 26, 19 : 27, 20 : 28,
                         21 : 29, 22 : 3, 23 : 30, 24 : 31, 25 : 32, 26 : 33, 27 : 34, 28 : 35, 29 : 36, 30 : 37,
	                 31 : 38, 32 : 39, 33 : 4, 34 : 40, 35 : 5, 36 : 6, 37 : 7, 38 : 8, 39 : 9}
                    
def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram
  
class Paadas(App):
  def build(self):
    model = tf.keras.models.load_model("marathi-40")
    #model.summary()

    x = tf.io.read_file('1_0.wav')
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
    x = tf.squeeze(x, axis=-1)
    waveform = x
    x = get_spectrogram(x)
    x = x[tf.newaxis,...]
    prediction = model(x)
    cls = np.argmax(prediction)
    print (class_to_number[cls])

if __name__ == '__main__':
    Paadas().run()
