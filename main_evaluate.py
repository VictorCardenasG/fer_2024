from classifier.config import cfg
from classifier.model import load_model
from evaluation.evaluation import predict_emotion
from evaluation.evaluation import traverse_and_predict
# Load the trained model
model = load_model(cfg, "C:/Users/Victor Cardenas/Documents/msc/semestre-3/idi_iii/fer_2024/models/model.pth")

# Example usage
main_folder = r"C:/Users/Victor Cardenas/Documents/MSC/SEMESTRE II/IDI II/PYTHON/ai_training_happy_sad_surprise"
results = traverse_and_predict(main_folder, model)

# Optional: Print summary of results
correct_predictions = sum(1 for _, true_emotion, predicted_emotion in results if true_emotion == predicted_emotion)
total_predictions = len(results)
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

print(f"Total images: {total_predictions}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}")