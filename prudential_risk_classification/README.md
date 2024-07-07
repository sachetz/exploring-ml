# Prudential Risk Classification

For decision trees, from the validation curve, the depth 12 was chosen as the optimal max depth for the learning curve.

For logistic regression, from the validation curve, the value 1 was chosen as the optimal C for the learning curve.

From the two learning curve graphs, we can observe that the generalization gap is higher for decision trees and lower for logistic regression. Also, the test score for the decision trees increases evenly until 4500 samples after which the increase becomes marginal, whereas for logistic regression the test score increases rapidly till around 2500 samples, after which the increase is marginal.

This indicates that logistic regression generalises well with a smaller number of samples, whereas decision trees take a larger number of samples to achieve the same scores. We notice that, in both cases, the training and test scores seem to converge as the number of samples increase, which indicates that both models would generalise well. However, the plateau or marginal increase in the test scores in both graphs indicate that the model performances with the chosen hyperparameters might be limited.

## Data

https://www.kaggle.com/c/prudential-life-insurance-assessment