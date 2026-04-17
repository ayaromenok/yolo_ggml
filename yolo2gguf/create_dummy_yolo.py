import torch
import torch.nn as nn

class DummyYOLO(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        return self.fc(self.conv1(x))

model = DummyYOLO()
# Simulate Ultralytics checkpoint format
checkpoint = {
    'epoch': 0,
    'model': model.state_dict(),
    'optimizer': None
}

torch.save(checkpoint, 'dummy_yolo.pt')
print("Created dummy_yolo.pt")
