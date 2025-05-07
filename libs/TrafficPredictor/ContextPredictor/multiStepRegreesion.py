import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
from typing import Union, List, Optional
from scipy.ndimage import uniform_filter1d

class MultiStepPredictor:
    def __init__(self, 
                 future_window_size: int = 10, 
                 predict_step: int=2,
                 Ts:float=0.01,
                 poly_degree: int = 3):
        self.L = future_window_size
        self.K = predict_step
        self.poly_degree = poly_degree
        self.Ts = Ts
    
    def _prepare_polynomial_features(self, X):
        poly = PolynomialFeatures(degree=self.poly_degree)
        return poly.fit_transform(X.reshape(-1, 1))
    
    def _predict_one_step(self, context) -> np.ndarray:
        # Prepare features and target for regression
        X = (self.Ts*np.arange(len(context))).reshape(-1, 1)
        y = context
        
        # Polynomial regression
        X_poly = self._prepare_polynomial_features(X)
        regressor = LinearRegression()
        regressor.fit(X_poly, y)
        
        # Predict future steps
        future_X =  (self.Ts*np.arange(len(context), len(context) + self.K)).reshape(-1, 1)
        future_X_poly = self._prepare_polynomial_features(future_X)
        predictions = regressor.predict(future_X_poly)
        
        return predictions
    
    def _smooth_predict_with_context(self, predictions, context) -> np.ndarray:
        """
        Smooth predictions using Savitzky-Golay filter
        """
        # Apply Savitzky-Golay filter for smoothing
        input = np.concatenate((context, predictions))
        #output = savgol_filter(input, len(input), self.poly_degree)
        output = uniform_filter1d(input, size=max(5, int(self.K/5)))
        return output[len(context):]

    def predict(self, context):
        context_with_predict = context
        predictOut = []
        N_step = int(np.floor(self.L/self.K))
        for i in range(N_step):
            predictions = self._predict_one_step(context_with_predict)           
            smoothedPredictions = self._smooth_predict_with_context(predictions, context_with_predict)
            context_with_predict = np.concatenate((context_with_predict[self.K:], smoothedPredictions))
            predictOut.append(smoothedPredictions)
        return np.concatenate(predictOut)