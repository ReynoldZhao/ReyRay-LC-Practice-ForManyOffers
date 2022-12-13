# Explanation

## Module:
Instead of using ResNet-18, we try a deeper neural network of ResNet-50, which may explore more potential features and hidden information in images.

## Data Augmentaion:
To enrich image data sources and promote the robustness of the model, we perform tranformation operation on images, like cropping, rotating and flipping, which prevents the model from overfitting

## Change of Predictions and Metrics:
The example ResNet model provided by TA uses label as the final prediction results. We assume this is  more complex because it directly predicts the lable "Poor" or "Rich", however, this label is computed from the combination of "Urban" label and "WealthPooled" value.

## Seperate Model targeted at Urban and Rural Dataset
We noticed that the final result of rich level depends on "Urban" label and "WealthPooled" value. Therefore, we train models on urban datasets and rural datasets seperately. Since the threshold of richness level and distribution on urban area and rural area are different. The independent training can enable model's specific learning ability on them.