from PIL import Image
import numpy as np
import torch
import training

image = Image.open('../image.png').convert('L')
image = image.resize((28, 28))
image = np.array(image) / 255.0
image = (image - 0.5) / 0.5
image = 1.0 - image

image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(training.device)

training.model.eval()
with torch.no_grad():
    output = training.model(image)
    prediction = torch.argmax(output, 1).item()

print(f"Predicted digit: {prediction}")