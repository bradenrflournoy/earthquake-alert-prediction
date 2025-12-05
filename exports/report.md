# ML Group 90 Project Proposal

**Literature Review**

Early warning systems for earthquakes are extremely important to mitigate the unfortunate impact of natural disasters. These systems provide time-sensitive alerts that give governments and communities time to prepare and act accordingly. The current conventional approaches to these systems mainly rely on just magnitude thresholds and seismic networks to trigger the early warning alerts. However, recent research involving machine learning and earthquake prediction has shown that adding features like depth, community-reported intensity, and instrumental measures to models can improve the accuracy of these systems. 

For example, T.T. Trugman, E.A. Casarottie, and P.M. Shearer [1] applied machine learning methods to earthquake records and determined that combining seismic features greatly improved the prediction of shaking intensity. Also, S. Mousavi and G. Beroza [2] found how neural networks and ensemble models using deep learning outperform the simple threshold-based systems in determining seismic events. These research papers, and many more, highlight the strong potential to improve the predictive performance of the earthquake alerts by using machine learning and emphasizing the importance of incorporating multiple features. 

**Problem Description**

Earthquakes are one of the most destructive natural disasters, often striking without warning and devastating communities with little time to spare. Currently, communities rely on early warning systems to provide alerts that allow them to prepare. However, many of these systems depend on simple thresholds of magnitude or seismic activity that fail to capture the full complexity of an earthquake and can lead to inaccurate alerts. This project aims to address these challenges by developing machine learning models that classify earthquake alert levels using a broader range of seismic features. By leveraging data from the Kaggle Earthquake Alert Prediction dataset, the goal is to improve the accuracy and reliability of these early warning systems to better protect these communities.

**Dataset Description**

The dataset contains 1,300 earthquake records with features such as

Magnitude: (Richter Scale), 
Depth: (km), 
CDI: The maximum reported intensity for the event range,
MMI: The maximum estimated instrumental intensity for the event,
Significance score: Based on magnitude, intensity, reports, and impact; higher values mean more significant,
Labeled alert category: Target Variable - either Green, Yellow, Orange, or Red.

**Dataset Link**

https://www.kaggle.com/datasets/ahmeduzaki/earthquake-alert-prediction-dataset?select=earthquake_alert_balanced_dataset.csv

**Data Preprocessing Methods**

Distribution stabilization: The cdi and mmi features are heavily left-skewed and stabilizing the variance will reduce this skew which helps with linear models and k-NN to learn better boundaries. “sklearn.preprocessing.PowerTransformer()”

Scale normalization: This will help ensure that all the loss function's penalties are fair to all features. depth and sig have significantly greater values than the other features and this normalization will ensure that the one feature isn’t dominating learning. “sklearn.preprocessing.StandardScaler()”

Class imbalance handling: This will ensure that misclassifying rare classes is penalized more than more common classes. This is important in our case because there are few “red” alerts which are the target variable. “sklearn.preprocessing.RandomOverSampler()”

Supervised feature selection: This will ensure that the noise and redundancies from some variables won’t impact learning. This will select features with the strongest class-conditional signal which will tighten decision boundaries and stabilize hyperparameters.“SelectKBest”

Our data preprocessing can be found in our GitHub repository under notebooks/Group90.ipynb.

**ML Models**

Logistic Regression: This provides a strong and interpretable baseline model for classification. It assumes linear decision boundaries, and its coefficients will also help indicate which features most strongly influence earthquake alert levels. "sklearn.linear_model.LogisticRegression"

Random Forest: This grouping of decision trees captures non-linear interactions among features such as magnitude, depth, and significance. It is robust to noise and reduces overfitting by averaging across multiple trees. "sklearn.ensemble.RandomForestClassifier"

Gradient Boosting: A boosted tree method such as XGBoost iteratively refines weak learners to achieve higher accuracy. This model is especially effective for structured tabular datasets. "xgboost.XGBClassifier"

For the midpoint we implemented Logistic Regression. Our implementation can be found in our GitHub repository under notebooks/Group90.ipynb.

**Results and Discussion**

**Quantitative Metrics Discussion**

The Logistic Regression model achieved strong and balanced performance across the four alert levels. Based on the quantitative metrics:

- Macro F1 Score (~0.79) shows that the model performs consistently across all classes, giving equal weight to each alert level (Green, Orange, Red).

- Balanced Accuracy (~0.79) shows the model predicts each class fairly well, even when the number of samples per class differs.

- Macro ROC-AUC (~0.93) shows the model can reliably distinguish between the different alert levels using probability scores.

- Brier Score (~0.09) shows the model’s probability estimates are well-calibrated and not overconfident.

The per-class metrics show that:

- “Green” and “Orange” alerts are predicted with high precision (0.76–0.91), meaning few false alarms.

- “Red” alerts have very high recall (0.92), meaning the model correctly catches most dangerous cases — this is ideal for an early-warning system where missing severe events is catastrophic.

Overall, these metrics indicate the model captures meaningful patterns between seismic features (magnitude, depth, CDI, MMI, significance) and alert level while maintaining balanced predictive performance.



The Random Forest model achieved very strong and stable performance across all alert levels. Based on the quantitative metrics:

- Macro F1 Score (~0.93) indicates excellent overall classification performance, showing that the model handles all four classes consistently well.

- Balanced Accuracy (~0.92) is high, meaning the model predicts each class with strong recall despite natural differences in class frequencies.

- Macro ROC-AUC (~0.99) shows exceptional ability to separate the alert levels using predicted probabilities.

- Macro Brier Score (~0.03) indicates extremely well-calibrated probabilities—Random Forest assigns confident predictions without being overconfident.

The per-class metrics highlight:

- Green, Orange, and Red alerts all achieved precision and recall values above ~0.95, showing very reliable classification on the majority of events.

- Yellow alerts also performed strongly with precision (~0.89) and recall (~0.92), representing a major improvement over Logistic Regression where Yellow recall was the main weakness.

Overall, the Random Forest model demonstrates that capturing non-linear interactions among features (magnitude, depth, CDI, MMI, significance) leads to significantly better predictive performance than the linear baseline. This model comfortably surpasses all project performance goals and produces highly trustworthy probability estimates.


The Gradient Boosting model achieved excellent overall performance across all four alert levels and consistently matched or exceeded the performance goals set in the project proposal. Based on the quantitative metrics:

- Macro F1 Score (~0.92) shows the model produces strong, balanced performance across all classes.

- Balanced Accuracy (~0.92) indicates equally strong recall for every alert level despite natural variations in the dataset.

- Macro ROC-AUC (~0.99) demonstrates that the model is highly effective at separating alert levels using probability scores.

- Macro Brier Score (~0.03) reflects extremely well-calibrated probabilities—comparable to Random Forest and significantly better than Logistic Regression.

The per-class results show:

- Green, Orange, and Red alerts all achieve very high precision (0.96–0.98) and recall (0.87–0.92), making the model reliable across the most common alert categories.

- Yellow alerts see major improvement relative to Logistic Regression, achieving precision ~0.86 and recall ~0.92, showing the model successfully handles the previously most challenging class.

Overall, Gradient Boosting provides strong, consistent, and well-calibrated performance. It effectively models the nonlinear interactions among earthquake features and achieves one of the most balanced and reliable sets of metrics among the three evaluated models.

**Visualizations Discussion**

For logistical regression, the confusion matrix shows that most predictions fall along the diagonal, meaning the model correctly classifies the majority of earthquakes. A few misclassifications occur between Orange and Red alerts — this makes sense since these alert levels are adjacent in severity and share similar ranges of magnitude and intensity features. The normalized confusion matrix also shows that recall (the proportion of correctly identified alerts per class) remains high for all alert levels.

<img width="1316" height="1156" alt="image" src="https://github.com/user-attachments/assets/84965d7d-0e5d-4bbd-aeed-82dc1b1bffd4" />
<img width="1312" height="1160" alt="image" src="https://github.com/user-attachments/assets/7319cfde-cd71-4e76-92a1-d7de3c70dc9f" />


The ROC curves shows the tradeoff between the true positive rate and false positive rate for each alert class. Each curve lies well above the diagonal “random guess” line, showing the model’s ability to separate each alert level effectively. The area under the curves (AUC) is around 0.85 on average, which supports the earlier quantitative metrics.

<img width="1584" height="1174" alt="image" src="https://github.com/user-attachments/assets/4fd494d4-e47e-4b49-a0ef-ae1de37a0920" />

The precision–recall curves reinforce the same conclusion: the model maintains good precision for frequent classes like Green and Orange, while achieving strong recall for Red, which is the most safety-critical.

<img width="1480" height="1184" alt="image" src="https://github.com/user-attachments/assets/424ced38-1c1b-417a-a34a-27e3f4b275bf" />



The Random Forest confusion matrix shows that nearly all predictions lie on the diagonal, meaning each alert level is classified correctly with very few mistakes. Misclassifications are sparse and mainly appear as small confusion between Green and Yellow or between Yellow and Orange, which is expected since these classes share overlapping ranges of shaking intensities.

<img width="591" height="578" alt="image" src="https://github.com/user-attachments/assets/a4adbcb2-22db-4f84-8cc4-69762e9c4afb" />
<img width="576" height="590" alt="image" src="https://github.com/user-attachments/assets/8dbf98c0-7f7d-4504-ad51-968090444bca" />

The ROC curves further show that every alert class achieves an AUC close to 1.00, indicating near-perfect separability. Each curve lies well above the diagonal, reflecting almost no overlap between positive and negative cases for each class in the one-vs-rest setting.

<img width="690" height="590" alt="image" src="https://github.com/user-attachments/assets/73ab36dc-847d-4260-b099-95d62528c84e" />

The precision–recall curves reinforce the same conclusion. All four alert levels maintain high precision and recall throughout most thresholds, with average precision values near 0.98–0.99. The Yellow class, which was the most challenging for the logistic regression model, achieves an AP of 0.95, showing that Random Forest corrects the earlier imbalance in performance.

<img width="690" height="590" alt="image" src="https://github.com/user-attachments/assets/976fb117-7d3d-4baa-9e84-04eb02a6dfcf" />


The confusion matrix for the Gradient Boosting model shows that most predictions fall along the diagonal, indicating that the model correctly classifies the majority of earthquake alerts. Misclassifications are minimal, with small confusion occurring between Green–Yellow and Orange–Yellow alerts—expected given their similar intensity profiles.

<img width="591" height="578" alt="image" src="https://github.com/user-attachments/assets/077d19b3-4059-4833-8f49-71fb95da9904" />
<img width="576" height="590" alt="image" src="https://github.com/user-attachments/assets/e868ab79-95bf-4999-a532-0cdc0ca6c9b6" />

The ROC curves show that all four classes achieve AUC values between 0.97 and 1.00, demonstrating near-perfect separability between the classes. Each curve lies far above the diagonal baseline, confirming that Gradient Boosting produces highly discriminative probability estimates.

<img width="690" height="590" alt="image" src="https://github.com/user-attachments/assets/98d67621-1697-44e3-9ad4-aaee582f2e78" />

These curves show that Gradient Boosting retains high precision even at high recall values. Although the Yellow class remains the most challenging, its AP score is still very strong, and substantially improved from the logistic regression baseline.

<img width="690" height="590" alt="image" src="https://github.com/user-attachments/assets/8b27d6c9-dbfe-46b9-848a-40d00a28faf8" />


**Analysis**

Our group aims to build a machine learning model that is able to classify earthquake alert levels (Green, Yellow, Orange, Red) using a range of seismic features (magnitude, depth, CDI, MMI, and significance) instead of relying on only magnitude. As a baseline, we implemented a logistic regression mode, following our plan in the project proposal. In our project proposal we set several key performance goals: a macro F1 score ≥ .75, a balanced accuracy ≥ .80, a macro ROC-AUC ≥ .85, a Brier score ≤ .12, and Red recall ≥ .80 with our most critical goal being the red recall due to it being most important to life safety.

Our logistic regression model successfully achieved or exceed most of these with values of:

- **Macro F1:** .786  
- **Macro ROC-AUC:** .929  
- **Macro Brier:** .086  

Our model’s recall for the Red class was .923 which as seen in the confusion matrix means that this model correctly identifies 60 of 65 Red events in the test set which far surpasses our minimum safety target. However our balanced accuracy came in at .788 which is just under our .80 goal. Balanced accuracy is the average of all class’s recall with each individual class’s recall being:

- **Green Recall:** .800  
- **Orange Recall:** .800  
- **Red Recall:** .923  
- **Yellow Recall:** .631  

While this model performed well on Red alerts and solid on the Green and Orange alerts, Yellow’s poor performance pulled the average down. The confusion matrix confirms this as many Yellow events were often misclassified as Green or Orange, suggesting that there is a complex relationship between the Green, Yellow, and Orange feature that this model struggles to show.

Overall we believe that this model’s strong performance is due to our preprocessing pipeline. The PowerTransformer normalized and skewed cdi and mmi features, while the StandardScaler ensured that features like depth and sig which had large ranges didn’t dominate our model. The GridSearchCV also selected PolynomialFeatures(degree=2), which confirmed that interactions between features are critical for separating classes.



Our second model builds on our baseline by using a Random Forest classifier. Unlike logistic regression, which relies on linear decision boundaries, Random Forests can pick up more complex patterns between the seismic features (magnitude, depth, CDI, MMI, and significance). This made it a strong next step for our project since the relationships between alert levels are unlikely to be strictly linear.

We used the same preprocessing steps as before - the PowerTransformer for CDI and MMI, and StandardScaler for all features. We then ran a 5-fold GridSearchCV over several parameters, including the number of trees, maximum depth, and split rules. The best model from this search used 100 trees with no limit on depth.

The Random Forest achieved strong and balanced performance with the following overall results:
* Accuracy: .923  
* Macro F1: .924  
* Macro Recall: .923  

Per-class recall also showed strong performance across all alert levels:
* Green Recall: .846  
* Orange Recall: .954  
* Red Recall: .969  
* Yellow Recall: .923  

These results show that the model identifies nearly all Red alerts, which remains our most important safety goal. It also performed especially well on Yellow and Orange alerts, suggesting that the nonlinear structure of Random Forests allows the model to pick up on subtle feature interactions that the logistic regression model struggled to capture.

The confusion matrix supports this, with most predictions falling along the diagonal and relatively few alerts being confused with neighboring levels. Misclassifications that do occur tend to be between adjacent severity levels, which is expected given their shared feature ranges.

Overall, the Random Forest results indicate that a more flexible model can better capture the relationships in our dataset. Its strong recall across all four alert classes, especially for Red and Yellow alerts, suggests that this approach may be better suited to representing the underlying structure of the seismic features.



With the XGBoost pipeline above:

Macro F1: 0.923  
Balanced accuracy: 0.923  
Macro ROC-AUC: 0.986  
Macro Brier: 0.031  
Per-class recall: Green: 0.892, Orange: 0.969, Red: 0.923, Yellow: 0.908

Gradient boosting:
- Meets or exceeds all project targets (macro F1 ≥ .75, balanced accuracy ≥ .80, macro ROC-AUC ≥ .85, Brier ≤ .12, Red recall ≥ .80).
- Maintains high recall for the critical Red alert level.
- Produces strong classification performance for the Yellow alert level.
- Achieves excellent probability calibration (Brier score ≈ 0.03).

Gradient Boosting (XGBoost) Results

To capture non-linear interactions between seismic features, we trained a gradient boosting model using XGBoost with the same preprocessing pipeline used throughout the project (PowerTransformer on CDI/MMI, StandardScaler, optional polynomial features, and RandomOverSampler for class balancing). The final model used 200 boosted trees with max depth 4 and learning rate 0.1.

On the held-out test set, XGBoost achieved a macro F1 of 0.923, a balanced accuracy of 0.923, a macro ROC–AUC of 0.986, and a macro Brier score of 0.031. These values satisfy all of our predefined project goals.

The per-class performance shows strong predictive ability across all alert levels. Recall for Green and Orange alerts reached 0.892 and 0.969, respectively, while Red recall remained high at 0.923. The Yellow class also showed strong performance, with recall of 0.908 and a corresponding F1 score near 0.89. These values indicate that the boosted trees are able to capture the complex boundaries between seismic patterns associated with different alert levels.

The confusion matrix for XGBoost shows that most examples lie on the diagonal, with only a small number of cross-class confusions. The normalized matrix highlights the consistently high recall across all four alert levels. ROC and precision–recall curves for each class lie well above the random baseline, with areas under the ROC curves close to 1.0, indicating strong separability of the seismic patterns that correspond to each alert level.

Overall, gradient boosting provides highly accurate and reliable earthquake alert classification across all severity levels. The model maintains high recall for the most severe Red alerts while also achieving strong performance for intermediate alert levels, making it a strong candidate for a practical early warning system.

**Model Comparison**

We were able to build three different Machine Learning models: linear regression, random forest, and gradient boosting. Based on the results we achieve, the gradient boosting model stood out as the best choice out of the three. This is because it provides the best balance of classification performance and reliability. The logistic repregression model was able to meet all of the minimum project targets but, because of its simplicity, it wasn't able to capture non-linear relationships. This, in turn, caused a low Yellow Recall score. When we utilized Random Forest, we were able to see an improvement from the linear regression model. This is because random forest was able to capture these non-linear interactions, achieving strong Marco F1 and Red Recall scores. However, the gradient boosting model was the most fitting out of the three since it scored higher on the Marco ROC-AUC and Brier scores. Based on these metrics, we can determine that the gradient boosting model ensures the highest accuracy and most trustworthy probability estimates.

**Next Steps**

While our current best performing model is the gradient boosting model, which has a very high recall on Red alerts, there are several directions and items we could explore that could further improve the system. The first thing to improve on in future work is using richer data and features. Incorporating additional seismological features such as station count/ density, distance to population centers, and regional indicators can better indicate and capture how an event translates into real world impact. Additionally we will analyze feature importance to better understand how variables such as magnitude, depth, CDI, and MMi affect and drive the different alert levels. 

Beyond adding features, we also plan to perform sensitivity studies to see how robust the model is to missing or noisy inputs. Finally, to move closer to a deployable early warning tool, we will also consider operational constraints such as inference speed and complexity. Comparing the runtime and resource requirements of gradient boosting to simpler baselines will help determine whether performance gains justify the added complexity in a real time earthquake alert system.


**Directory Guide**
/data/ : This contains our data set

/data/earthquake_alert_balanced_data.csv :  This is our data set

/notebooks/ : This contains our Jupyter Notebook file.

/notebooks/Group90.ipynb : This contains our data preprocessing methods, ML algorithm, and visualizations.

/.ipynb_checkpoints/ : This folder is extra and should be ignored.



<!-- <img width="1290" height="658" alt="image" src="https://github.com/user-attachments/assets/1cfd64d7-ad4e-431e-8b2a-ced63509442a" /> -->

![Dashboard screenshot](contributions_table.png)

![Dashboard screenshot](/GANTT_CHART.png)





**References**

[1] T. T. Trugman, E. A. Casarotti, and P. M. Shearer, “Machine learning for earthquake early warning and ground motion prediction,” Seismological Research Letters, vol. 91, no. 5, pp. 2362–2376, 2020.
[2] S. Mousavi and G. C. Beroza, “A machine-learning approach for earthquake magnitude estimation,” Geophysical Research Letters, vol. 47, no. 1, 2020.
[3] X. Wang, Z. Wang, J. Wang, P. Miao, H. Dang, and Z. Li, “Machine learning based ground motion site amplification prediction,” Frontiers in Earth Science, vol. 11, 2023.
