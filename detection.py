import numpy as np


class AccidentDetectionModel(object):
    # index 0 = Accident, index 1 = Non Accident
    # Must match training class order: {'Accident': 0, 'Non Accident': 1}
    class_nums = ['Accident', 'Non Accident']

    def __init__(self, model_json_file, model_weights_file):
        # ── Keras 3 compatible loading ──────────────────────
        # model_from_json was removed in Keras 3.
        # We use tf.keras.models.model_from_json via TensorFlow directly.
        try:
            # Try Keras 3 / TF 2.12+ approach first
            import tensorflow as tf
            with open(model_json_file, "r") as json_file:
                loaded_model_json = json_file.read()
            self.loaded_model = tf.keras.models.model_from_json(loaded_model_json)

        except Exception as e:
            # Fallback for older TF versions
            from tensorflow.keras.models import model_from_json
            from tensorflow.keras.models import Sequential
            with open(model_json_file, "r") as json_file:
                loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(
                loaded_model_json,
                custom_objects={"Sequential": Sequential}
            )

        # Load weights — supports both .h5 and .weights.h5
        self.loaded_model.load_weights(model_weights_file)

    def predict_accident(self, img):
        self.preds = self.loaded_model.predict(img, verbose=0)
        predicted_class = AccidentDetectionModel.class_nums[np.argmax(self.preds)]
        return predicted_class, self.preds