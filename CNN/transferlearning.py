from keras.preprocessing.image import ImageDataGenerator
from keras.applications import densenet
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras import regularizers
from keras import backend as K
from keras_applications.inception_v3 import InceptionV3, preprocess_input
import os

#Get the count of number of files in this folder and all subfolders
def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])

#Get count of number of subfolders directly below the folder in path
def get_num_subfodlers(path):
    if not os.path.exists(path):
        return 0
    return sum([len(d) for r, d, files in os.walk(path)])

def create_img_generator():
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

#Main Code
ImageWidth, ImageHeight = 299, 299
Training_epochs = 100
Batch_size = 32
Number_of_FC_neurons = 1024

train_dir = 'D:\\DB\\vehicle\\train'
test_dir = 'D:\\DB\\vehicle\\test'

num_train_samples = get_num_files(train_dir)
num_classes = get_num_subfodlers(train_dir)
num_test_samples = get_num_files(test_dir)

num_epochs = Training_epochs
batch_size = Batch_size

train_img_generator = create_img_generator()
test_img_generator = create_img_generator()

train_generator = train_img_generator.flow_from_directory(
    train_dir,
    target_size=(ImageWidth, ImageHeight),
    batch_size=batch_size,
    seed=50
)

validation_generator = test_img_generator.flow_from_directory(
    test_dir,
    target_size=(ImageWidth, ImageHeight),
    batch_size=batch_size,
    seed=50
)

InceptionV3_base_model = InceptionV3(weights='imagenet', include_top=False)
print('Inception V3 is loaded')

x = InceptionV3_base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(Number_of_FC_neurons, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs = InceptionV3_base_model.input, outputs=predictions)

print(model.summary())

print("Transfer Learning")

for layer in InceptionV3_base_model.layers:
    layer.trainable = False

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=num_train_samples// batch_size,
    validation_data=validation_generator,
    validation_steps=num_test_samples // batch_size,
    class_weight='auto'
)

model.save('transportation_100_epochs.h5')

#Found 519164 images belonging to 11 classes.