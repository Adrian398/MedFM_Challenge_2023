# Pseudo code
# todo compare best aggregate model with ensemble predict
# Let's say these are the accuracies of each model on each class.
# accuracy[model_index][class_index] = accuracy_of_model_on_class
accuracy = [
    [0.9, 0.8, 0.85, 0.7],  # Model 1
    [0.88, 0.9, 0.8, 0.85],  # Model 2
    [0.7, 0.88, 0.88, 0.9],  # Model 3
    [0.8, 0.7, 0.9, 0.85],  # Model 4
    [0.85, 0.82, 0.8, 0.8],  # Model 5
]


def ensemble_predict(models, x):
    """
    Make a weighted prediction for an input x using the ensemble of models.

    Args:
    - models: List of models
    - x: input sample

    Returns:
    - Final weighted prediction
    """

    total_predictions = [0, 0, 0, 0]

    for i, model in enumerate(models):
        predictions = model.predict(x)  # assuming the prediction is a probability distribution

        for j, pred in enumerate(predictions):
            total_predictions[j] += pred * accuracy[i][j]

    # Normalize the predictions so they sum up to 1
    sum_predictions = sum(total_predictions)
    normalized_predictions = [pred / sum_predictions for pred in total_predictions]

    return normalized_predictions

# To use the function:
# predictions = ensemble_predict(models, input_sample)
