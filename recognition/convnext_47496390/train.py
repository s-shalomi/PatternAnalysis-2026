import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")


# Training for Baseline CNN

from modules import BaselineCNN, convnext_base, convnext_tiny
from dataset import build_dataloaders

CLASS_NAMES = ["NC", "AD"]
learning_rate = 1e-4
batch_size = 64
num_epochs = 300
weight_decay = 0.05


def create_model(model_name, num_classes=2):
    model_name = model_name.lower()

    if model_name == "cnn":
        model = BaselineCNN(num_classes=num_classes)
        # 1 input channel (grayscale) -> prep images during training/testing
        expected_in_channels = 1
        rgb = False
    else:
        model = convnext_tiny(num_classes=num_classes)
        # 3 input channels -> prep images
        expected_in_channels = 3
        rgb = True

    return model.to(device), expected_in_channels, rgb

def train_epoch(model, loader, criterion, optimizer, expected_channels):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = 100 * correct/total

    return epoch_loss, epoch_accuracy

def validate(model, loader, criterion, expected_channels):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

        loss = running_loss / total
        accuracy = 100.0 * correct / total
        return loss, accuracy

def test(model, loader, criterion, expected_channels):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []
    misclassified_samples = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Get predicted classes
            predictions = outputs.argmax(dim=1) # predicted class index for each sample
            probs = torch.softmax(outputs, dim=1) # probability distribution across all classes for each sample
            confidences = probs.max(dim=1).values #  probability assigned to the predicted class for each sample

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            correct += (predictions == labels).sum().item()
            total += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probs.cpu().tolist())

            # Collect misclassified samples (store on CPU to save GPU memory)
            incorrect_mask = predictions != labels
            if incorrect_mask.any():
                bad_images = images[incorrect_mask].cpu()
                bad_preds = predictions[incorrect_mask].cpu().tolist()
                bad_trues = labels[incorrect_mask].cpu().tolist()
                bad_confs = confidences[incorrect_mask].cpu().tolist()

                for i in range(len(bad_preds)):
                    # Cap collection at 16 to avoid bloating RAM
                    if len(misclassified_samples) < 16:
                        misclassified_samples.append({
                            "image": bad_images[i],
                            "pred": bad_preds[i],
                            "true": bad_trues[i],
                            "confidence": bad_confs[i],
                        })

    test_loss = running_loss / total
    test_accuracy = 100.0 * correct / total
    
    print("\n--- Test Results ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
    ))
    
    print("Confusion Matrix:")
    print(confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1],
    ))

    return (
        test_loss,
        test_accuracy,
        all_labels,
        all_predictions,
        all_probabilities,
        misclassified_samples,
    )

def pipeline(model_name="convnext"):
    model, expected_channels, rgb_gray = create_model(model_name)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    train_loader, val_loader, test_loader = build_dataloaders(
        rgb_gray,  # grayscale or rgb depending on model
        data_root="/home/groups/comp3710/ADNI/AD_NC",
        batch_size=batch_size
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_epochs = []

    best_val_accuracy = -1.0

    print("\nTraining model...")

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            expected_channels,
        )
        scheduler.step()

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Accuracy: {train_accuracy:.2f}%"
        )

        # validate every 2nd epoch
        if (epoch + 1) % 2 == 0:
            val_loss, val_accuracy = validate(
                model,
                val_loader,
                criterion,
                expected_channels,
            )

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_epochs.append(epoch + 1)

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.2f}%"
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), f"{model_name}_best.pth")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        
    end_time = time.perf_counter() 
    print(f"Total training and validation time: {end_time - start_time:.2f} seconds")

    # Testing
    print("Testing model")

    # load best model
    model.load_state_dict(torch.load(f"{model_name}_best.pth", map_location=device))

    (
        test_loss,
        test_accuracy,
        all_labels,
        all_predictions,
        all_probabilities,
        misclassified_samples,
    ) = test(
        model,
        test_loader,
        criterion,
        expected_channels,
    )

    return (
        train_losses,
        train_accuracies,
        val_losses,
        val_accuracies,
        test_loss,
        test_accuracy,
        val_epochs,
        all_labels,
        all_predictions,
        all_probabilities,
        misclassified_samples,
    )
    
    
def train_test_baseline():
    #model = BaselineCNN(num_classes=2).to(device)
    model = convnext_tiny(num_classes=2).to(device)

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

    # for plotting
    val_epochs = []

    best_val_accuracy = 0

    print("Training baseline model...")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    
    start_time = time.perf_counter()

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
                val_running_loss = 0
                val_total = 0
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_running_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                
                val_loss = (val_running_loss / len(val_loader))
                val_accuracy = (100.0 * val_correct / val_total)
                
                val_accuracies.append(val_accuracy)
                val_losses.append(val_loss)

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy

                    torch.save(model.state_dict(), "baseline_best.pth")

                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.2f}%"
                )
            
            val_epochs.append(epoch + 1) # for plotting

        
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        
    end_time = time.perf_counter() 
    print(f"Total training and validation time: {end_time - start_time:.2f} seconds")

    # Testing
    print("Testing model")

    # load best model
    model.load_state_dict(torch.load("baseline_best.pth", map_location=device))

    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []  # needed for ROC curve
    misclassified_samples = [] # for false positives
    

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            # Get predicted classes
            predictions = outputs.argmax(dim=1) # predicted class index for each sample
            probs = torch.softmax(outputs, dim=1) # probability distribution across all classes for each sample
            confidences = probs.max(dim=1).values #  probability assigned to the predicted class for each sample

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probs.cpu().tolist())

            # Collect misclassified samples (store on CPU to save GPU memory)
            incorrect_mask = predictions != labels
            if incorrect_mask.any():
                bad_images = images[incorrect_mask].cpu()
                bad_preds = predictions[incorrect_mask].cpu().tolist()
                bad_trues = labels[incorrect_mask].cpu().tolist()
                bad_confs = confidences[incorrect_mask].cpu().tolist()

                for i in range(len(bad_preds)):
                    # Cap collection at 16 to avoid bloating RAM
                    if len(misclassified_samples) < 16:
                        misclassified_samples.append({
                            "image": bad_images[i],
                            "pred": bad_preds[i],
                            "true": bad_trues[i],
                            "confidence": bad_confs[i],
                        })

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

    return (train_losses, train_accuracies, val_losses, val_accuracies, test_loss, test_accuracy, 
        val_epochs, all_labels, all_predictions, all_probabilities, misclassified_samples)


#if __name__ == "__main__":
#    train_test_baseline()


