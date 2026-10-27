import torch
import torch.nn as nn

from src.model import VisionTransformer, epochs, learning_rate


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main():
    from src.data import train_loader, val_loader

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Training on {device}")
    model = VisionTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.2%} | "
            f"Validation loss: {val_loss:.4f}, accuracy: {val_accuracy:.2%}"
        )


if __name__ == "__main__":
    main()

