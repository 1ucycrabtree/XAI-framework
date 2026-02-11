import pandas as pd

class Dataset:
    def __init__(
        self,
        X : pd.DataFrame,
        y : pd.Series | pd.DataFrame | None = None,
        target_label=None,
        feature_names=None,
        metadata=None
    ):
        self.X = X
        self.y = y
        self.target_label = target_label
        self.feature_names = feature_names or list(X.columns)
        self.metadata = metadata or {}
        
        self._validate()
        
    def _validate(self):
        if not isinstance(self.X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        if self.X.shape[0] == 0:
            raise ValueError("Dataset X cannot be empty")
        
        if set(self.feature_names) != set(self.X.columns):
            raise ValueError("Feature names must match the columns in X")
        
        if self.y is not None:
            if not isinstance(self.y, (pd.Series, pd.DataFrame)):
                raise ValueError("y must be a pandas Series or DataFrame")
            
            if len(self.y) != len(self.X):
                raise ValueError("Length of y must match the number of rows in X")
            
            if not self.X.index.equals(self.y.index):
                raise ValueError("Index of X and y must match")
        
        if not isinstance(self.feature_names, list) or not all(isinstance(f, str) for f in self.feature_names):
            raise ValueError("Something has gone wrong parsing the X columns to features. Feature names must be a list of strings.")
        
        if set(self.feature_names) != set(self.X.columns):
            raise ValueError("Feature names must match the columns in X")
        
        if self.metadata and not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be a dictionary")