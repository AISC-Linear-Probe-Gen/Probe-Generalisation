import torch
import joblib
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

Scaler = StandardScaler | MaxAbsScaler | MinMaxScaler | RobustScaler

class LogRegProbe:
    def __init__(self, **kwargs):
        model_kwargs = {k: v for k, v in kwargs.items() if k in SGDClassifier().get_params()}
        self.model = SGDClassifier(loss="log_loss", **model_kwargs)
        self.scaler: Scaler = kwargs.get('scaler', StandardScaler())
        self.needs_scaling = False
        self.classes = None

    def to_numpy(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return value

    def fit(self, X: torch.Tensor, y: torch.Tensor, scale: bool = True):
        X = self.to_numpy(X)
        y = self.to_numpy(y)

        if scale:
            X = self.scaler.fit_transform(X)
            self.needs_scaling = True
        self.model.fit(X, y)
        self.classes = np.unique(y)

    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, classes=None, scale: bool = True):
        X = self.to_numpy(X)
        y = self.to_numpy(y)

        if scale:
            if hasattr(self.scaler, "partial_fit"):
                self.scaler.partial_fit(X)
                X = self.scaler.transform(X)
            else:
                X = self.scaler.fit_transform(X)
            self.needs_scaling = True

        if self.classes is None:
            if classes is None:
                classes = np.unique(y)
            self.classes = np.array(classes)
            self.model.partial_fit(X, y, classes=self.classes)
        else:
            self.model.partial_fit(X, y)

    def predict(self, X: torch.Tensor):
        X = self.to_numpy(X)
        if self.needs_scaling:
            X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def save(self, path: str):
        joblib.dump(
            {
                'model': self.model,
                'scaler': self.scaler,
                'needs_scaling': self.needs_scaling,
                'classes': self.classes,
            },
            path,
        )

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.needs_scaling = data['needs_scaling']
        self.classes = data['classes']
