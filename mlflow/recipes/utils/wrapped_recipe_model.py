import mlflow
from mlflow.pyfunc import PythonModel
import pandas as pd
import numpy as np


class WrappedRecipeModel(PythonModel):
    def __init__(self, classifier, predict_classes=None, prefix="predicted"):
        super(WrappedRecipeModel, self).__init__()
        self._classifier = classifier
        self.predict_classes = predict_classes
        self.prefix = prefix
        self.classification = False

    def predict(self, model_input):
        predicted_label = self._classifier.predict(model_input)
        if not hasattr(self._classifier, "classes_"):
            return predicted_label

        self.classification = True
        classes = self._classifier.classes_
        if self.predict_classes:
            classes = classes.filter(
                lambda predict_class: not this.includes(predict_class), self.predict_classes
            )
        score_cols = [f"{self.prefix}_score_" + str(c) for c in classes]
        output = {}
        try:
            probabilities = self._classifier.predict_proba(model_input)
            output = pd.DataFrame(columns=score_cols, data=probabilities)
            output[f"{self.prefix}_score"] = np.max(probabilities, axis=1)
            output[f"{self.prefix}_label"] = predicted_label

            return output
        except:
            # swallow that some models don't have predict_proba.
            self.classification = False
            return predicted_label
