
---

# Music Genre Classifier

This project involves building a deep learning model to classify music genres from audio clips using TensorFlow and Keras. The classifier uses mel spectrograms as input features.

## Requirements

- TensorFlow
- Keras
- numpy
- scipy
- matplotlib
- pydub

## Setup

1. **Dataset:** Ensure your dataset is organized in directories by genre under `../DATASET/sesler/Data/genres_original/`.

2. **Preprocessing:**
   - `load_wav_16k_mono(filename)`: Loads a WAV file as mono 16kHz.
   - `audio_to_melspectogram(wav, sr=16000, n_mels=128, n_fft=2048, hop_length=512)`: Converts audio to mel spectrogram.

3. **Data Preparation:**
   - Load and preprocess audio files from the dataset.
   - Concatenate and prepare datasets for training.

4. **Model Training:**
   - Define and train a Convolutional Neural Network (CNN) model using TensorFlow and Keras.
   - Model architecture:
     - Input layer: (28, 128, 1)
     - Convolutional layers with MaxPooling
     - Dense layers with dropout
     - Output layer with softmax activation (11 classes)
   - Optimizer: Adam with custom parameters.
   - Loss function: Categorical Crossentropy.

5. **Model Evaluation:**
   - Evaluate the model on test data.
   - Plot training history using matplotlib.
   - ![alt text](https://github.com/[HayatiYrtgl]/[Music_Genre_Classification]/blob/[master]/music_classifier.png?raw=true)

6. **Save Model:**
   - Save the trained model as `Music_Classifier.h5`.

7. **Audio Prediction:**
   - Use `split_audio(file_path)` to split an MP3 file into chunks.
   - Preprocess chunks into mel spectrograms for prediction.
   - Load the saved model and predict genres for each chunk.

## Usage

- Ensure dependencies are installed (`pip install -r requirements.txt`).
- Run scripts sequentially for dataset preparation, model training, and audio prediction.

---
