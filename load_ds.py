import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
import ollama
from PIL import Image
import io
import matplotlib.pyplot as plt
import pickle

def one_hot_concepts(concepts):
    I = np.unique(concepts)
    one_hots = []
    diag_matrix = np.eye(len(I))
    for sample in range(len(concepts)):
        for i in range(len(I)):
            if concepts[sample] == I[i]:
                one_hots.append(diag_matrix[i])
    one_hots = np.stack(one_hots)
    return one_hots

def load_s3d(base_path = "C:/Users/debryu/Desktop/VS_CODE/HOME/ML/Tirocinio/interpreter/data/shapes3d/3dshapes.h5"):
  print('Loading the dataset...')
  print(base_path)
  with h5py.File(base_path, 'r') as f:
    images = f['images']
    concepts = f['labels']

    images   = images[()]
    concepts = concepts[()]
    labels   = np.copy(concepts)

  #Labels
  predictions = []
  for j in range(labels.shape[0]):
      if labels[j,4] == 3 and labels[j,2] == 0.0:
          predictions.append(1)
      else:
          predictions.append(0)

  labels = np.array(predictions)

  #Concepts
  one_hots_to_concat = []
  concepts = concepts[:,:5] # Remove the orientation concept
  shape = concepts.shape
  for i in range(shape[1]):
      colunm = concepts[:,i]
      one_hots_to_concat.append(one_hot_concepts(colunm))
  preprocess_concepts = np.hstack(one_hots_to_concat)

  return images, preprocess_concepts, labels


class SHAPES3D(torch.utils.data.Dataset):
    def __init__(self, split='train', triad = [], args=None):

      self.images, self.concepts, self.labels = load_s3d()
      self.transform = transforms.Compose(
          [transforms.ToTensor(),
          #transforms.Resize((224,224)) #EDIT
          ]   
      )
      self.resize = transforms.Compose(
          [transforms.ToTensor(),
            transforms.Resize((224,224))]
      )
      # self.concept_mask=np.array([False]*len(self.list_images))

        
    def __getitem__(self, idx):
        image = self.images[idx]
        concepts = self.concepts[idx]
        labels = self.labels[idx]
        return image, labels, concepts

    def __len__(self):
        return len(self.images)


ds = SHAPES3D()
dl = torch.utils.data.DataLoader(
                                  ds,
                                  shuffle=True,
                                  batch_size=3,
                                  #num_workers=num_workers,   # temporary fix
                                  drop_last=True,
                                  )
print("dataset loaded")
new_ds = []
for image, concept, label in dl:
  image = image[1]
  concept = concept[1]
  label = label[1]
  print(image.shape, concept.shape, label.shape)
  #plt.imshow(image)
  #plt.show()
  image = image.numpy()
  img = Image.fromarray(image)
  img_byte_arr = io.BytesIO()
  img.save(img_byte_arr, format='PNG')
  llama_img = img_byte_arr.getvalue()
  img_byte_arr.close()
  messages = [
      {"role": "user", "content": "What is depicted inside the image?", "images": [llama_img]},
    ]
  response = ollama.chat(
    model = "x/llama3.2-vision",
    messages = messages
  )
  messages.append(response)
  print(response["message"]['role'])
  print(response["message"]['content'])
  
  template = "[Object color: yellow\nObject shape: cube\nWall color: green\nFloor color: purple\n]"
  template += "[Object color: light blue\nObject shape: pill\nWall color: light green\nFloor color: orange\n]"
  template += "[Object color: pink\nObject shape: cylinder\nWall color: green\nFloor color: red\n]"
  m = {"role": "user", "content": 'Please provide me with the annotations of the image, following this examples:' + template}

  messages.append(m)
  response = ollama.chat(
    model = "x/llama3.2-vision",
    messages = messages
  )
  messages.append(response)
  annotations = response["message"]['content']

  sample = {'img':image, 'concept':concept, 'label':label, 'annotations':annotations}
  new_ds.append(sample)
  print(response["message"]['role'])
  print(response["message"]['content'])
  if len(new_ds) == 10000:
    break

ds_len = len(new_ds)
pickle.dump(new_ds, open(f"C:/Users/debryu/Desktop/VS_CODE/HOME/ML/work/askLLAMA/data/llama_ds/ds_{ds_len}.pkl", "wb"))

    

