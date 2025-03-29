def calculate_metrics(tp, fp, tn, fn):
    """
    Calculate precision, recall, F1 score, and accuracy from confusion matrix values.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Accuracy": accuracy
    }

def calculate_confusion_matrix(predictions, ground_truth, crime_classes):
    """
    Calculate TP, FP, TN, FN for crime detection in CCTV frames.
    
    Args:
        predictions (list of lists): Model predictions for each frame (e.g., [["weapon"], ["person"], ...])
        ground_truth (list of lists): Actual labels for each frame (e.g., [["weapon"], [], ...])
        crime_classes (set): Classes indicating crime (e.g., {"weapon", "suspicious"})
    
    Returns:
        dict: TP, FP, TN, FN counts
    """
    tp, fp, tn, fn = 0, 0, 0, 0

    for pred, truth in zip(predictions, ground_truth):
        # Check if crime is predicted (True/False)
        pred_positive = any(obj in crime_classes for obj in pred)
        # Check if crime is actually present (True/False)
        truth_positive = any(obj in crime_classes for obj in truth)

        if pred_positive and truth_positive:
            tp += 1
        elif pred_positive and not truth_positive:
            fp += 1
        elif not pred_positive and truth_positive:
            fn += 1
        else:
            tn += 1

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}

# Example Usage
if __name__ == "__main__":
    # Define crime-related classes (customize based on your dataset)
    crime_classes = {"weapon", "suspicious"}

    # Example Model Predictions (one sublist = one frame's detected objects)
    predictions = [
        ["weapon", "person"],  # Frame 1: Crime detected (TP)
        ["person"],            # Frame 2: No crime (TN)
        ["suspicious"],        # Frame 3: Crime detected (FP if ground truth has no crime)
        [],                    # Frame 4: No crime (TN)
    ]

    # Actual Ground Truth (one sublist = one frame's true objects)
    ground_truth = [
        ["weapon", "person"],  # Frame 1: Crime exists (TP)
        ["person"],            # Frame 2: No crime (TN)
        [],                    # Frame 3: No crime (FP)
        [],                    # Frame 4: No crime (TN)
    ]

    # Calculate confusion matrix
    confusion_matrix = calculate_confusion_matrix(predictions, ground_truth, crime_classes)
    tp = confusion_matrix["TP"]
    fp = confusion_matrix["FP"]
    tn = confusion_matrix["TN"]
    fn = confusion_matrix["FN"]

    # Calculate metrics
    metrics = calculate_metrics(tp, fp, tn, fn)

    # Print results
    print("Confusion Matrix:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}\n")

    print("Performance Metrics:")
    print(f"Precision: {metrics['Precision']:.2f}")
    print(f"Recall: {metrics['Recall']:.2f}")
    print(f"F1 Score: {metrics['F1 Score']:.2f}")
    print(f"Accuracy: {metrics['Accuracy']:.2f}")
