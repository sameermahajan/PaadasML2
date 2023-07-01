import tensorflow as tf
from keras import models
import numpy as np

class_to_number = {0 : 1, 1 : 10, 2 : 11, 3 : 12, 4 : 13, 5 : 14, 6 : 15, 7 : 16, 8 : 17, 9 : 18, 10 : 19,
	          11 : 2, 12 : 21, 13 : 22, 14 : 23, 15 : 24, 16 : 25, 17 : 26, 18 : 27, 19 : 28, 20 : 29}
                    
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
