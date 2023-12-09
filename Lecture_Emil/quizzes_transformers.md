1. **Question:** What is the main function of transformers in the scikit-learn library?
   <details><summary>Click for Answer</summary>
   Transformers in scikit-learn are primarily used for preprocessing data, such as scaling, normalizing, encoding categorical variables, or performing other transformations on data before it is used in machine learning models.
   </details>

2. **Question:** Why is it important to use transformers within a pipeline in scikit-learn?
   <details><summary>Click for Answer</summary>
   Using transformers within a pipeline ensures consistency in data transformations, facilitates the easy replication of preprocessing steps, and helps in preventing data leakage during model training and validation.
   </details>

3. **Question:** Can you name a scenario where a custom transformer would be necessary in scikit-learn?
   <details><summary>Click for Answer</summary>
   A custom transformer is necessary when there is a need for a specific data transformation that is not available in the existing scikit-learn transformers. This could be a specific feature engineering step or a unique way of handling data.
   </details>

4. **Question:** How does a transformer in scikit-learn handle categorical data differently from numerical data?
   <details><summary>Click for Answer</summary>
   Transformers in scikit-learn typically handle categorical data by encoding it into numerical values using techniques like one-hot encoding, label encoding, etc., while numerical data might be scaled or normalized.
   </details>

5. **Question:** What is the significance of the `fit_transform` method in scikit-learn transformers?
   <details><summary>Click for Answer</summary>
   The `fit_transform` method in scikit-learn transformers combines the fitting and transformation steps into one call, which is more efficient as it reduces the computational overhead of performing these operations separately.
   </details>

6. **Question:** True or False: Transformers in scikit-learn can only handle numerical data.
   <details><summary>Click for Answer</summary>
   False. Transformers in scikit-learn can handle both numerical and categorical data, depending on the specific transformer used.
   </details>

7. **Question:** Select the correct transformer in scikit-learn that is used for normalizing data: a) `MinMaxScaler` b) `Normalizer` c) `StandardScaler`
   <details><summary>Click for Answer</summary>
   b) `Normalizer`
   </details>

8. **Question:** True or False: A pipeline in scikit-learn can contain multiple transformers but only one estimator.
   <details><summary>Click for Answer</summary>
   True. A pipeline can have multiple transformers for preprocessing steps, but it typically ends with a single estimator for making predictions.
   </details>

9. **Question:** Which of the following methods is not typically available in a scikit-learn transformer? a) `fit` b) `transform` c) `predict`
   <details><summary>Click for Answer</summary>
   c) `predict`
   </details>

10. **Question:** True or False: In scikit-learn, it's mandatory to use the `fit_transform` method for all transformers.
   <details><summary>Click for Answer</summary>
   False. While `fit_transform` is often more efficient, it's not mandatory; the `fit` and `transform` methods can be used separately.
   </details>

11. **Question:** Write a code snippet to demonstrate how to integrate a `StandardScaler` and a `PCA` transformer into a scikit-learn pipeline.
    <details><summary>Click for Answer</summary>
    
    ```python
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2))
    ])
    ```
    </details>

12. **Question:** How would you modify the parameters of a `MinMaxScaler` inside a pipeline named `my_pipeline` to have a feature range between 0 and 1?
    <details><summary>Click for Answer</summary>
    
    ```python
    my_pipeline.set_params(minmaxscaler__feature_range=(0, 1))
    ```
    </details>
