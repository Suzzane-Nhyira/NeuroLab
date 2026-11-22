
from models.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


y_true = [0, 0, 1, 1, 1, 0, 1, 0]

y_pred = [0.1, 0.3, 0.8, 0.7, 0.4, 0.6, 0.9, 0.2]


print("NEUROLAB EVALUATION TEST")
print("-" * 35)

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print()

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
