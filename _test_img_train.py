"""End-to-end image training test using synthetic images."""
import torch, torch.nn as nn, torch.optim as optim, time
from pathlib import Path
from PIL import Image
import numpy as np

# Create synthetic dataset: 3 classes, 30 images each
print('Creating synthetic image dataset...')
base = Path('_test_images')
for cls, color in [('red', (200,50,50)), ('green', (50,200,50)), ('blue', (50,50,200))]:
    (base / cls).mkdir(parents=True, exist_ok=True)
    for i in range(30):
        arr = np.full((64, 64, 3), color, dtype=np.uint8)
        arr += np.random.randint(-30, 30, arr.shape, dtype=np.int16).clip(0,255).astype(np.uint8)
        Image.fromarray(arr).save(base / cls / f'img_{i:03d}.jpg')
print('Created 90 synthetic images (3 classes x 30)')

# Test image_dataset
from image_dataset import build_image_loaders
train, val, class_names, info = build_image_loaders(
    [str(base)], img_size=32, batch_size=16)
print(f'Info: {info}')

xb, yb = next(iter(train))
print(f'Batch: x={xb.shape} y={yb.shape} classes={class_names}')

# Train for 3 epochs
from implementations import HMTImageClassifier
model = HMTImageClassifier(num_classes=3, dim=64, patch_size=8,
                            num_layers=2, num_heads=4, num_scales=2)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

opt     = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1, 4):
    model.train()
    total = correct = n = 0
    for xb, yb in train:
        opt.zero_grad()
        out  = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total   += loss.item()
        correct += (out.argmax(1) == yb).sum().item()
        n       += yb.size(0)
    print(f'Epoch {epoch}: loss={total/len(train):.4f}  acc={correct/n*100:.1f}%')

# Save checkpoint
import json
from datetime import datetime
Path('trained_models').mkdir(exist_ok=True)
ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
name = f'ImageClassifier_{ts}'
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {'model_type':'Image Classification','hidden_dim':64,'num_layers':2,
               'epochs':3,'batch_size':16,'lr':0.001,'optimizer':'Adam',
               'scheduler':'None','num_heads':4,'reflector':False},
    'data_info': info,
    'class_names': class_names,
    'model_arch': {'type':'HMTImageClassifier','num_classes':3,'dim':64,
                   'num_layers':2,'patch_size':8,'img_size':32},
}, f'trained_models/{name}.pt')
with open(f'trained_models/{name}.json','w') as f:
    json.dump({'name':name,'model_type':'Image Classification','accuracy':'see above',
               'status':'ready','task':'image_classification','class_names':class_names,
               'img_size':32,'weights_file':f'trained_models/{name}.pt'}, f, indent=2)
print(f'Saved: trained_models/{name}.pt')

# Test inference via model_chat
from model_chat import _find_model, _load_meta, _detect_task, ImageSession
pt   = _find_model(name)
ckpt = torch.load(str(pt), map_location='cpu', weights_only=False)
meta = _load_meta(pt)
task = _detect_task(ckpt, meta)
print(f'Detected task: {task}')
assert task == 'image_classification', f'Expected image_classification, got {task}'

session = ImageSession(pt, ckpt, meta)
# Classify one test image
session._classify_one(str(base / 'red' / 'img_000.jpg'))
session._classify_one(str(base / 'blue' / 'img_000.jpg'))

print('ALL TESTS PASSED')

# Cleanup
import shutil
shutil.rmtree(str(base))
