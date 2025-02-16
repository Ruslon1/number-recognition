import torch
import torch.nn as nn
import torch.optim as optim
import cnn
import dataset

model = cnn.CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cpu')

for epoch in range(1):
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

torch.save(model.state_dict(), '../model.pth')