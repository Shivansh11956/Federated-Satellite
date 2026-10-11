import flwr as fl
import tensorflow as tf
from model import create_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 64

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    "/content/dataset_part2/",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    "/content/dataset_part2/",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model = create_model()

class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)

        model.fit(
            train_data,
            epochs=5,
            validation_data=val_data,
            verbose=1
        )

        return model.get_weights(), train_data.samples, {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)

        loss, accuracy = model.evaluate(val_data, verbose=0)

        return loss, val_data.samples, {"accuracy": float(accuracy)}

fl.client.start_numpy_client(
    server_address="koqma-2409-40c4-9-d006-7dbb-bf0a-fb0-79ab.run.pinggy-free.link:46073",
    client=FlowerClient()
)