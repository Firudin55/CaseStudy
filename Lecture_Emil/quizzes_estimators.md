1. **Question:** What is the primary purpose of an estimator in scikit-learn?
   <details><summary>Click for Answer</summary>
   The primary purpose of an estimator in scikit-learn is to build a model from data. Estimators are used for classification, regression, or clustering tasks and are trained using labeled data to make predictions or identify patterns.
   </details>

2. **Question:** How does a supervised learning estimator differ from an unsupervised learning estimator in scikit-learn?
   <details><summary>Click for Answer</summary>
   A supervised learning estimator requires labeled data for training and learns to predict outcomes based on input features. In contrast, an unsupervised learning estimator works with unlabeled data and identifies patterns or structures within the data, like clustering or dimensionality reduction.
   </details>

3. **Question:** What role does the `fit` method play in the context of an estimator in scikit-learn?
   <details><summary>Click for Answer</summary>
   The `fit` method is used to train the estimator on a given dataset. It involves the estimator learning from the provided data, adjusting its parameters to model the underlying pattern or relationship in the training data.
   </details>

4. **Question:** Can you explain the difference between a parametric and a non-parametric estimator in scikit-learn?
   <details><summary>Click for Answer</summary>
   A parametric estimator makes assumptions about the form of the function that maps inputs to outputs and learns a finite set of parameters (e.g., linear regression). Non-parametric estimators, on the other hand, do not make explicit assumptions about this function and can adapt to a wider range of data shapes (e.g., decision trees).
   </details>

5. **Question:** Why is it important to evaluate an estimator's performance on unseen data in scikit-learn?
   <details><summary>Click for Answer</summary>
   Evaluating an estimator's performance on unseen data is crucial to assess its generalization ability. It helps in understanding how well the model will perform on new, real-world data, beyond the examples it was trained on.
   </details>

6. **Question:** True or False: Estimators in scikit-learn are only used for predictive modeling.
   <details><summary>Click for Answer</summary>
   False. While many estimators in scikit-learn are used for predictive modeling, others are used for tasks like clustering and dimensionality reduction, which are not strictly predictive.
   </details>

7. **Question:** Which scikit-learn estimator is typically used for classification tasks? a) `LinearRegression` b) `RandomForestClassifier` c) `PCA`
   <details><summary>Click for Answer</summary>
   b) `RandomForestClassifier`
   </details>

8. **Question:** True or False: The `predict` method can be used with all types of estimators in scikit-learn.
   <details><summary>Click for Answer</summary>
   False. The `predict` method is specific to supervised learning estimators. Unsupervised learning estimators might not have a `predict` method (e.g., PCA).
   </details>

9. **Question:** Select the correct statement: In scikit-learn, a) `fit` method is used to predict outcomes, b) `transform` method is used to train the model, c) `predict` method is used to generate predictions.
   <details><summary>Click for Answer</summary>
   c) `predict` method is used to generate predictions.
   </details>

10. **Question:** True or False: It's necessary to always normalize data before using any estimator in scikit-learn.
   <details><summary>Click for Answer</summary>
   False. While normalization can be beneficial for many algorithms, it's not an absolute requirement for all estimators in scikit-learn.
   </details>

11. **Question:** Write a Python snippet to train a `RandomForestRegressor` on a dataset `X_train` with target values `y_train`.
    <details><summary>Click for Answer</summary>
    
    ```python
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    ```
    </details>

12. **Question:** How can you access the feature importances of a trained `RandomForestClassifier` named `my_classifier`?
    <details><summary>Click for Answer</summary>
    
    ```python
    feature_importances = my_classifier.feature
    feature_importances = my_classifier.feature_importances_
    ```
    </details>