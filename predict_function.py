from pydub import AudioSegment
import numpy as np
from preprocess_the_audio import audio_to_melspectogram, tf, resample
from keras.models import load_model
from train_test_data import test_dataset


def split_audio(file_path, chunk_length_ms=30000):
    audio = AudioSegment.from_mp3(file_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks


# Örnek kullanım

classes = ['pop', 'metal', 'blues', 'classical', 'disco', 'rock', 'reggae', 'hiphop', 'country', 'jazz']


def load_wav_from_audiosegment(audio_segment):
    """Convert AudioSegment to 16k mono wav."""
    samples = np.array(audio_segment.get_array_of_samples())
    samples = resample(samples, 16000)
    return tf.convert_to_tensor(samples, dtype=tf.float32)


def preprocess_chunk(audio_segment):
    """Convert audio segment to mel spectrogram."""
    wav = load_wav_from_audiosegment(audio_segment)
    spectrogram = audio_to_melspectogram(wav)
    spectrogram = tf.expand_dims(spectrogram, -1)
    return spectrogram


mp3_file_path = 'musics/Late Night Mood Jazz   Relaxing Smooth Jazz   Saxophone By ONSTAGE Band.mp3'
chunks = split_audio(mp3_file_path)
spectrograms = [preprocess_chunk(chunk) for chunk in chunks]

model = load_model('Music_Classifier.h5')
model.evaluate(test_dataset)

# Tahmin yapma
predictions = [model.predict(tf.expand_dims(spec, 0)) for spec in spectrograms]


# Sonuçları yazdırma
for i, prediction in enumerate(predictions):
    print(f"Chunk {i + 1} prediction: {classes[np.argmax(prediction[0])]}")
