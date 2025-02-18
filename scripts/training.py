import torch
import torch.nn as nn
import torch.optim as optim
import cnn
import dataset
import os

model = cnn.CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cpu')

checkpoint_path = '../checkpoint.pth'

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}.")
else:
    start_epoch = 0
    print("Starting from scratch")

for epoch in range(start_epoch, start_epoch):
    model.train()
    running_loss = 0.0
    for images, lables in dataset.train_loader:
        images, lables = images.to(device), lables.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, lables)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataset.train_loader):.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': running_loss
    }, checkpoint_path)
""" 
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in dataset.test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        print("  " + str(labels.size(0)) + "  " + str(predicted))
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%") """