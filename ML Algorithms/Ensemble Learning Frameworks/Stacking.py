from sklearn.model_selection import cross_val_predict
from sklearn.base import clone, BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin

class StackingEnsemble(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, base_models, meta_model, use_features_in_secondary=False):
        self.base_models = base_models
        self.meta_model = meta_model
        self.use_features_in_secondary = use_features_in_secondary
    
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        
        # Train base models and create the new feature set for the meta model
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            preds = cross_val_predict(model, X, y, cv=5, method="predict")
            self.base_models_[i].append(model.fit(X, y))
            meta_features[:, i] = preds
        
        # Train the meta model
        if self.use_features_in_secondary:
            self.meta_model_.fit(np.hstack((X, meta_features)), y)
        else:
            self.meta_model_.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        
        if self.use_features_in_secondary:
            return self.meta_model_.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_model_.predict(meta_features)
