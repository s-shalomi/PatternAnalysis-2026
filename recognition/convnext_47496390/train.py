import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")


# Training for Baseline CNN

from modules import BaselineCNN
from dataset import build_dataloaders

learning_rate = 0.001
batch_size = 32
num_epochs = 10

def train_test_baseline():
    model = BaselineCNN(num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    train_loader, val_loader, test_loader = build_dataloaders(
        False,  # grayscale model, not RGB 
        data_root="/home/groups/comp3710/ADNI/AD_NC",
        batch_size=batch_size
    )

    total_step = len(train_loader)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    print("Training baseline model...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_step
        epoch_accuracy = 100 * correct/total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {epoch_loss:.4f}, "
            f"Train Accuracy: {epoch_accuracy:.2f}%"
        )

        # validate every 2nd epoch
        if (epoch+1) % 2 == 0: 
            model.eval()
            with torch.no_grad():
                val_correct = 0
                val_loss = 0
                val_total = 0
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                val_loss = (val_running_loss / len(val_loader))
                val_accuracy = (100.0 * val_correct / val_total)
                
                val_accuracies.append(val_accuracy)
                val_losses.append(val_loss)

                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Val Loss: {val_epoch_loss:.4f}, "
                    f"Val Accuracy: {val_epoch_accuracy:.2f}%"
                )

    # Testing
    print("Testing model")
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            # Get predicted classes
            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    # Calculate metrics
    test_loss /= len(test_loader)
    test_accuracy = 100.0 * correct / total

    print("\n--- Test Results ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=["NC", "AD"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions, labels=[0, 1]))




if __name__ == "__main__":
    train_test_baseline()


