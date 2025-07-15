'''
import tensorflow as tf
from tensorflow.keras.applications import Xception, NASNetLarge, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 10
base_path = "./Final Training Data/"

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    base_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    base_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

def build_and_train_model(base_model, model_name):
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"Training {model_name}...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        verbose=1
    )
    
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f"{model_name} Validation Accuracy: {val_accuracy * 100:.2f}%")
    return val_accuracy

accuracies = {}
models = {
    "Xception": Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    "NASNetLarge": NASNetLarge(weights='imagenet', include_top=False, input_shape=(331, 331, 3)),
    "SENet": tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    "DenseNet121": DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
}

for model_name, model_base in models.items():
    accuracies[model_name] = build_and_train_model(model_base, model_name)

print("\nFinal Validation Accuracies:")
for model_name, accuracy in accuracies.items():
    print(f"{model_name}: {accuracy * 100:.2f}%")
'''

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 10
base_path = "./Final Training Data/"

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    base_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    base_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

def build_and_train_model(base_model, model_name):
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"Training {model_name}...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        verbose=1
    )
    
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f"{model_name} Validation Accuracy: {val_accuracy * 100:.2f}%")
    return val_accuracy

accuracies = {}
models = {
    "Xception": Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
}

for model_name, model_base in models.items():
    accuracies[model_name] = build_and_train_model(model_base, model_name)

print("\nFinal Validation Accuracies:")
for model_name, accuracy in accuracies.items():
    print(f"{model_name}: {accuracy * 100:.2f}%")
