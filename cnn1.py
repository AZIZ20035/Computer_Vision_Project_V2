import os
import shutil
import random
import tensorflow as tf
import keras.preprocessing.image as ke

# Paths
source_dir = './data/correct'  # Directory for "correct" images
train_dir = './data/train'
test_dir = './data/test'
correct_train_dir = os.path.join(train_dir, 'correct')
correct_test_dir = os.path.join(test_dir, 'correct')

# Create directories if they don't exist
os.makedirs(correct_train_dir, exist_ok=True)
os.makedirs(correct_test_dir, exist_ok=True)

# List all images in the source directory
all_images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Shuffle the list of images
random.shuffle(all_images)

# Define the split ratio
split_ratio = 0.8
split_index = int(len(all_images) * split_ratio)

# Split the images into training and test sets
train_images = all_images[:split_index]
test_images = all_images[split_index:]

# Move images to the correct directories for training and testing
for img in train_images:
    src_path = os.path.join(source_dir, img)
    dst_path = os.path.join(correct_train_dir, img)
    shutil.move(src_path, dst_path)

for img in test_images:
    src_path = os.path.join(source_dir, img)
    dst_path = os.path.join(correct_test_dir, img)
    shutil.move(src_path, dst_path)

# Ensure directories exist
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"The training directory '{train_dir}' does not exist")

if not os.path.exists(test_dir):
    raise FileNotFoundError(f"The test directory '{test_dir}' does not exist")

# Initialize ImageDataGenerator for training and test sets
train_datagen = ke.ImageDataGenerator(rescale=1./255,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True)
test_datagen = ke.ImageDataGenerator(rescale=1./255)

# Load training set
training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

# Load test set
test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# Check if data is loaded correctly
if training_set.samples == 0:
    raise ValueError("No training data found. Ensure the data directory is correctly structured.")
if test_set.samples == 0:
    raise ValueError("No test data found. Ensure the data directory is correctly structured.")

# Print the number of samples in each set
print(f"Number of training samples: {training_set.samples}")
print(f"Number of test samples: {test_set.samples}")

# Build the CNN model
cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN
history = cnn.fit(x=training_set, validation_data=test_set, epochs=45)

# Save the model
cnn.save('dumbbell_curl_model_updated.keras')

# Print train and test accuracy
train_accuracy = history.history['accuracy'][-1]
test_accuracy = history.history['val_accuracy'][-1]

print(f"Final training accuracy: {train_accuracy}")
print(f"Final test accuracy: {test_accuracy}")
