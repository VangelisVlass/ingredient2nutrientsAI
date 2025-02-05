import numpy as np

def softmax_normalization(values):
    """
    Applies Softmax normalization row-wise to a matrix.
    If all values in a row are zero, it keeps them as zero.
    """
    values = np.array(values)

    # Identify rows where all nutrient values are zero
    row_sums = np.sum(values, axis=1, keepdims=True)
    all_zero_rows = (row_sums == 0).flatten()  # Ensure it's a 1D boolean array

    # Create output array
    normalized_values = np.zeros_like(values)

    # Apply Softmax only to rows that are NOT all-zero
    non_zero_rows = ~all_zero_rows
    if np.any(non_zero_rows):  # Only apply if there are non-zero rows
        exp_values = np.exp(values[non_zero_rows] - np.max(values[non_zero_rows], axis=1, keepdims=True))
        normalized_values[non_zero_rows] = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    return normalized_values



def inverse_softmax(predictions):
    """
    Applies inverse Softmax by dividing each predicted value
    by the sum of predictions to recover proportions.
    """
    predictions = np.array(predictions)
    return predictions / np.sum(predictions, axis=1, keepdims=True)