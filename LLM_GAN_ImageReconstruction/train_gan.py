import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, LeakyReLU, Flatten, Reshape, UpSampling2D
from tensorflow.keras.models import Model
import cv2
import numpy as np
import os

def generate_chessboard(image_size=256, missing_squares=2):
    """Generate a synthetic chessboard with missing squares."""
    board = np.zeros((image_size, image_size), dtype=np.uint8)
    square_size = image_size // 8

    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                cv2.rectangle(board, (j * square_size, i * square_size),
                              ((j+1) * square_size, (i+1) * square_size), 255, -1)

    # Remove random squares
    for _ in range(missing_squares):
        x, y = np.random.randint(0, 8, size=2)
        board[y * square_size: (y+1) * square_size, x * square_size: (x+1) * square_size] = 0

    return board

# Generate and save sample training images
for i in range(100):  # Create 100 samples
    cv2.imwrite(f"dataset/secluded/chessboard_{i}.png", generate_chessboard(missing_squares=3))
    cv2.imwrite(f"dataset/complete/chessboard_{i}.png", generate_chessboard(missing_squares=0))

print("Synthetic chessboard dataset created successfully! ✅")

def preprocess_image(image_path, size=(256, 256)):
    """Preprocess chessboard images: Resize & Normalize."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size) / 255.0  # Normalize
    return image

# Load dataset
input_images = []  # Secluded boards
target_images = []  # Complete boards

for filename in os.listdir("dataset/secluded/"):
    img = preprocess_image(f"dataset/secluded/{filename}")
    input_images.append(img)

for filename in os.listdir("dataset/complete/"):
    img = preprocess_image(f"dataset/complete/{filename}")
    target_images.append(img)

input_images = np.array(input_images).reshape(-1, 256, 256, 1)
target_images = np.array(target_images).reshape(-1, 256, 256, 1)

# Build Generator
def build_generator():
    input_noise = Input(shape=(100,))
    x = Dense(64 * 64 * 128)(input_noise)
    x = Reshape((64, 64, 128))(x)
    x = UpSampling2D()(x)
    x = Conv2D(128, kernel_size=3, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D()(x)
    x = Conv2D(64, kernel_size=3, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(1, kernel_size=3, padding="same", activation="sigmoid")(x)

    return Model(input_noise, x)

# Build Discriminator
def build_discriminator():
    input_image = Input(shape=(256, 256, 1))
    x = Conv2D(64, kernel_size=3, strides=2, padding="same")(input_image)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dense(1, activation="sigmoid")(x)

    return Model(input_image, x)

# Compile GAN Model
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

discriminator.trainable = False  # Freeze discriminator in GAN training
gan_input = Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)

gan = Model(gan_input, gan_output)
gan.compile(optimizer="adam", loss="binary_crossentropy")

import matplotlib.pyplot as plt

epochs = 5000 
batch_size = 64

for epoch in range(epochs): # Train Discriminator idx = np.random.randint(0, input_images.shape[0], batch_size) real_images = target_images[idx] noise = np.random.normal(0, 1, (batch_size, 100)) fake_images = generator.predict(noise)

    # Train Discriminator
    idx = np.random.randint(0, input_images.shape[0], batch_size)
    real_images = target_images[idx]
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_images = generator.predict(noise)
    
    d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
    d_loss = np.add(d_loss_real, d_loss_fake) / 2

    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    # Display Progress
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

        # Save example generated image
        generated_image = generator.predict(np.random.normal(0, 1, (1, 100))).reshape(256, 256)
        plt.imshow(generated_image, cmap="gray")
        plt.savefig(f"generated_chessboard_{epoch}.png")
    
from tensorflow.keras.models import save_model

generator.save("/content/gan_chessboard_model.h5")  # Save model in Colab

    