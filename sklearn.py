import numpy as np


# QuantileTransformer
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(n_quantiles=100, output_distribution="normal", random_state=42)
train_target = qt.fit_transform(np.array(train_target).reshape(-1, 1))
prediction = qt.inverse_transform(model.predict(features_df[feature_names]).reshape(-1, 1)).ravel()
