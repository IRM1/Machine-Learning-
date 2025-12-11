import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix



(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Original train images shape:", x_train.shape)
print("Original train labels shape:", y_train.shape)
print("Original test images shape:", x_test.shape)
print("Original test labels shape:", y_test.shape)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def show_sample_images(x, y, class_names):
    plt.figure(figsize=(6, 6))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x[i])
       
        label_index = int(y[i][0])
        plt.title(class_names[label_index])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0


x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42
)

print("Train set shape:", x_train.shape, y_train.shape)
print("Validation set shape:", x_val.shape, y_val.shape)
print("Test set shape:", x_test.shape, y_test.shape)


model = models.Sequential([

    layers.Conv2D(
        32, (3, 3),
        activation="relu",
        padding="same",
        input_shape=(32, 32, 3)
    ),
 
    layers.MaxPooling2D((2, 2)),

  
    layers.Conv2D(
        64, (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),
  
    layers.Conv2D(
        128, (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])


print("\nModel summary:")
model.summary()




model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)



EPOCHS = 15 
BATCH_SIZE = 64

print("\nTraining starts")
start_time = time.time()


history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val)
)

end_time = time.time()

elapsed_seconds = end_time - start_time
print(f"Total training time: {elapsed_seconds:.2f} seconds (~{elapsed_seconds/60:.2f} minutes)")

print("Training completed\n")



def plot_history(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Acc")
    plt.plot(epochs_range, val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

# TESTING THE MODEL

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Get predictions for test set
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test.flatten()

print("\nClassification Report (per class):")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))


cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix shape:", cm.shape)



num_classes = cm.shape[0]

TP = np.diag(cm)

total_true = cm.sum(axis=1)

FN = total_true - TP

total_pred = cm.sum(axis=0)

FP = total_pred - TP

x = np.arange(num_classes)  
width = 0.25                 

plt.figure(figsize=(12, 6))

plt.bar(x - width, TP, width, label="True Positives",  color="tab:green")
plt.bar(x,         FN, width, label="False Negatives", color="tab:red")
plt.bar(x + width, FP, width, label="False Positives", color="tab:gray")

plt.xticks(x, class_names, rotation=45)
plt.ylabel("Number of test images")
plt.xlabel("CIFAR-10 class")
plt.title("Per-class True Positives, False Negatives, and False Positives")
plt.legend()
plt.tight_layout()
plt.show()

print("End of script.")