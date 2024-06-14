import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Enable TensorFlow with Metal on macOS
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("TensorFlow with Metal enabled.")
    except RuntimeError as e:
        print(e)

# Configure TensorFlow to use all available CPU cores
num_cores = len(tf.config.experimental.list_physical_devices('CPU'))
print(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)
tf.config.threading.set_intra_op_parallelism_threads(num_cores)

def preprocessing_for_training():
    path = './data/Training/'
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_generator = train_datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    return train_generator

def preprocessing_for_testing_inference(batchSize):
    path = './data/Testing/'
    test_val_datagen = ImageDataGenerator(rescale=1.0/255)
    test_generator = test_val_datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=batchSize,
        class_mode='categorical',
        shuffle=False
    )
    return test_generator

def build_model(hp):
    model = Sequential()
    for i in range(1, 3):
        model.add(Conv2D(
            filters=hp.Int(f'conv_{i}_filters', min_value=32, max_value=256, step=32),
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            input_shape=(224, 224, 3) if i == 1 else None
        ))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(Dense(
            units=hp.Int(f'dense_{i}_units', min_value=128, max_value=512, step=128),
            activation='relu'
        ))
        if hp.Boolean(f'dropout_{i}'):
            model.add(Dropout(rate=hp.Float(f'dropout_{i}_rate', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(4, activation='softmax'))
    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# Initialize the tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='./model_runs/',
    project_name='brain_tumor_classification'
)

# Search for the best hyperparameters
tuner.search(preprocessing_for_training(), epochs=50, validation_data=preprocessing_for_testing_inference(32))

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal hyperparameters are as follows:
Number of filters in each convolutional layer:
    Conv_1_filters: {best_hps.get('conv_1_filters')}
    Conv_2_filters: {best_hps.get('conv_2_filters')}
    Conv_3_filters: {best_hps.get('conv_3_filters')}
    Conv_4_filters: {best_hps.get('conv_4_filters')}
Number of dense layers: {best_hps.get('num_dense_layers')}
""")
for i in range(best_hps.get('num_dense_layers')):
    print(f"Dense_{i}_units: {best_hps.get(f'dense_{i}_units')}")
    print(f"Dropout_{i}_rate: {best_hps.get(f'dropout_{i}_rate')}")
print(f"Learning rate: {best_hps.get('learning_rate')}")

# Build the model with the optimal hyperparameters and train it
# model = tuner.hypermodel.build(best_hps)
# history = model.fit(preprocessing_for_training(), epochs=50, validation_data=preprocessing_for_testing_inference())
