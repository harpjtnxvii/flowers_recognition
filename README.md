# Flower Recognition Using CNN

Sebuah project deep learning untuk mengklasifikasikan gambar bunga menggunakan Convolutional Neural Network (CNN) yang dibangun dengan TensorFlow/Keras. Model ini bisa mengenali lima jenis bunga yang berbeda dengan akurasi yang cukup tinggi.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Overview

Project ini mengimplementasikan sistem klasifikasi gambar menggunakan CNN untuk mengenali jenis bunga dari foto. Modelnya di-train dengan dataset yang berisi lima kategori bunga dan menghasilkan accuracy yang lumayan bagus melalui teknik data augmentation dan regularization.

**Key Features:**
- Multi-class flower classification (5 categories)
- Data augmentation untuk meningkatkan generalisasi model
- Early stopping supaya gak overfitting
- Model checkpointing untuk save best weights
- Visualization dari training metrics

## Dataset

Dataset berisi gambar dari lima jenis bunga yang diorganisir dalam struktur folder seperti ini:

```
flowers/
├── daisy/
├── dandelion/
├── roses/
├── sunflowers/
└── tulips/
```

**Data Split:**
- Training: 80%
- Validation: 20%

Dataset otomatis di-split menggunakan `ImageDataGenerator` dengan parameter validation split.

## Installation

### Requirements
```bash
pip install tensorflow numpy matplotlib
```

### Setup
1. Clone repository ini
```bash
git clone https://github.com/yourusername/flower-recognition-cnn.git
cd flower-recognition-cnn
```

2. Mount Google Drive (kalau pakai Google Colab)
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Update path directory data di code
```python
data_dir = '/path/to/your/flowers/dataset'
```

## Project Structure

```
flower-recognition-cnn/
├── README.md
├── train.py              # Script untuk training
├── predict.py            # Script untuk prediction
├── model.py              # Architecture model
├── best_model_flowers.h5 # Saved model weights
└── notebooks/
    └── exploration.ipynb # Data exploration
```

## Model Architecture

CNN model terdiri dari beberapa layer sebagai berikut:

**Convolutional Layers:**
- Conv2D (32 filters, 3×3) + ReLU + MaxPooling
- Conv2D (64 filters, 3×3) + ReLU + MaxPooling
- Conv2D (128 filters, 3×3) + ReLU + MaxPooling

**Dense Layers:**
- Flatten layer
- Dense (128 units) + ReLU
- Dropout (0.5)
- Dense (5 units) + Softmax

**Compilation:**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy

Arsitekturnya straightforward tapi cukup effective untuk task klasifikasi bunga ini. Dropout layer ditambahkan untuk mencegah overfitting pada training data.

## Usage

### Training the Model

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load data
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Train model
history = model.fit(
    train_generator,
    epochs=40,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint]
)
```

### Making Predictions

Untuk melakukan prediksi pada gambar baru, bisa gunakan code berikut:

```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Load dan preprocess image
img_path = 'path/to/test/image.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
class_idx = np.argmax(pred)
labels = list(train_generator.class_indices.keys())

print(f"Prediction: {labels[class_idx]}")
print(f"Confidence: {pred[0][class_idx]:.2%}")
```

**Example Output:**
```
Prediction: sunflowers
Confidence: 92.34%
```

Pretty straightforward kan? Tinggal load image, preprocess, terus predict deh.

## Results

Model menunjukkan performa yang cukup bagus dengan characteristics sebagai berikut:

- **Training Strategy:** Early stopping dengan patience 8 epochs
- **Best Model:** Disimpan berdasarkan validation accuracy
- **Regularization:** Dropout (0.5) untuk prevent overfitting

### Training Metrics

Training history menunjukkan convergence yang bagus dari accuracy dan loss curves:

```python
# Visualize training progress
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

Dari graph yang dihasilkan, kita bisa lihat bahwa model tidak overfitting dan training process berjalan dengan smooth.

## Future Improvements

Ada beberapa hal yang bisa dikembangkan lebih lanjut:

- **Transfer Learning:** Implement pre-trained models seperti VGG16, ResNet50, atau EfficientNet untuk improved accuracy
- **Dataset Expansion:** Collect lebih banyak diverse flower images supaya generalization lebih bagus
- **Web Application:** Deploy pakai Flask atau Streamlit untuk user-friendly interface
- **Mobile Deployment:** Convert model ke TensorFlow Lite untuk Android/iOS applications
- **Additional Classes:** Expand ke lebih banyak flower species
- **Real-time Detection:** Implement live camera feed classification

Kalau mau lebih serious, transfer learning is the way to go karena bisa boost accuracy significantly.

## Contributing

Contributions are welcome! Kalau mau contribute, feel free untuk submit Pull Request. Untuk major changes, please open issue dulu untuk discuss apa yang mau diubah.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow/Keras documentation dan tutorials yang super helpful
- Dataset source: [Specify your dataset source here]
- Inspiration dari berbagai CNN image classification projects

## Notes

Project ini dibuat sebagai learning project untuk memahami CNN architecture dan image classification. Masih banyak room for improvement, tapi overall sudah cukup functional untuk recognize flower types dengan decent accuracy.

Kalau ada questions atau suggestions, jangan ragu untuk reach out atau open issue di repository ini!

---

**Author:** Harbangan Panjaitan 
**Last Updated:** November 26, 2025
