import torch
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

Scaler = StandardScaler | MaxAbsScaler | MinMaxScaler | RobustScaler

class LogRegProbe:
    def __init__(self, streaming: bool = False, **kwargs):
        self.streaming = streaming
        if streaming:
            model_kwargs = {k: v for k, v in kwargs.items() if k in SGDClassifier().get_params()}
            self.model = SGDClassifier(loss="log_loss", **model_kwargs)
        else:
            model_kwargs = {k: v for k, v in kwargs.items() if k in LogisticRegression().get_params()}
            self.model = LogisticRegression(**model_kwargs)
        self.scaler: Scaler = kwargs.get('scaler', StandardScaler())
        self._needs_scaling = False
        self._classes = None

    def fit(self, X: torch.Tensor, y: torch.Tensor, scale: bool = True):
        if self.streaming:
            self.partial_fit(X, y, scale=scale)
            return
        if scale:
            X = self.scaler.fit_transform(X)
            self._needs_scaling = True
        self.model.fit(X, y)

    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, classes=None, scale: bool = True):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        if scale:
            if hasattr(self.scaler, "partial_fit"):
                self.scaler.partial_fit(X)
                X = self.scaler.transform(X)
            else:
                X = self.scaler.fit_transform(X)
            self._needs_scaling = True

        if self._classes is None:
            if classes is None:
                classes = np.unique(y)
            self._classes = np.array(classes)
            self.model.partial_fit(X, y, classes=self._classes)
        else:
            self.model.partial_fit(X, y)

    def predict(self, X: torch.Tensor):
        if self._needs_scaling:
            X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def save(self, path: str):
        joblib.dump(
            {
                'model': self.model,
                'scaler': self.scaler,
                '_needs_scaling': self._needs_scaling,
                'streaming': self.streaming,
                '_classes': self._classes,
            },
            path,
        )

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self._needs_scaling = data['_needs_scaling']
        self.streaming = data.get('streaming', False)
        self._classes = data.get('_classes', None)
