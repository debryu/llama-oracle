from model import CBM
import torch
import pickle
import torchvision.transforms as transforms 
import numpy as np

data = pickle.load(open("C:/Users/debryu/Desktop/VS_CODE/HOME/ML/work/askLLAMA/data/llama_ds/ds_10000_one_hot.pkl", "rb"))
n_c = data[0]['annotations'].shape[0]


pre_data = []
for i in data:
  if type(i['annotations']) == str:
    continue
  else:
    pre_data.append(i)

data = pre_data

train_data = data[:8000]
test_data = data[8000:]



class Dataset(torch.utils.data.Dataset):
  def __init__(self, data):
    self.data = data
    self.transform = transforms.Compose(
      [transforms.ToTensor(),
      ]   
    )

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    sample = self.data[idx]
    img = sample['img']
    concept = sample['concept']
    label = sample['label']
    annotations = sample['annotations']
    return self.transform(img), concept, label, annotations

print(n_c)

model = CBM(n_c).cuda()
train_ds = Dataset(train_data)
test_ds = Dataset(test_data)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

epoch = range(100)
patience = 6
best_loss = np.inf

for e in epoch:
  train_loss = []
  model.train()
  for batch in train_dl:
    img, label, concept, annotations = batch
    img = img.cuda()
    label = label.cuda()
    concept = concept.cuda()
    annotations = annotations.cuda()
    #print(img.shape, concept.shape, label.shape, annotations.shape)
    z, rec = model(img)
    loss = model.loss(z, annotations)
    train_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print(f"Epoch {e} - Train Loss: {np.mean(train_loss)}")

  test_loss = []
  model.eval()
  for batch in test_dl:
    img, label, concept, annotations = batch
    img = img.cuda()
    label = label.cuda()
    concept = concept.cuda()
    annotations = annotations.cuda()
    z, rec = model(img)
    loss = model.loss(z, annotations)
    test_loss.append(loss.item())
  print(f"Epoch {e} - Test Loss: {np.mean(test_loss)}")
  if np.mean(test_loss) < best_loss:
    best_loss = np.mean(test_loss)
    best_model = model.state_dict()
    patience = 6
  else:
    patience -= 1
  
  if patience == 0:
    break

# Save the model
torch.save(best_model, "C:/Users/debryu/Desktop/VS_CODE/HOME/ML/work/askLLAMA/model.pth")