import pickle
import torch


ds_file = "C:/Users/debryu/Desktop/VS_CODE/HOME/ML/work/askLLAMA/data/llama_ds/ds_10000.pkl"
new_ds_file = "C:/Users/debryu/Desktop/VS_CODE/HOME/ML/work/askLLAMA/data/llama_ds/ds_10000_one_hot.pkl"
ds = pickle.load(open(ds_file, "rb"))

# First, get all the possible concept values

colors = set()
shapes = set()

for sample in ds:
  annotations = sample['annotations']
  if annotations[0] != "[":
    continue

  #print(annotations)
  annotations = annotations.replace("[","").replace("]","")
  #print(annotations)
  c = annotations.split("\n")
  if len(c) >= 7:
    continue
  for i in c:
    if len(i) <= 4:
      c.remove(i)
  
  #print("\n\n")
  #print(c)
  c = c[:4]
  obj_color,obj_shape,wall_color,floor_color = c
  obj_color = obj_color.split(":")[1].strip()
  obj_shape = obj_shape.split(":")[1].strip()
  wall_color = wall_color.split(":")[1].strip()
  floor_color = floor_color.split(":")[1].strip()
  colors.add(obj_color)
  colors.add(wall_color)
  colors.add(floor_color)
  shapes.add(obj_shape)

print(colors)
print(shapes)

colors = list(colors)
shapes = list(shapes)

new_ds = []
for sample in ds:
  obj_color = torch.zeros(len(colors))
  obj_shape = torch.zeros(len(shapes))
  wall_color = torch.zeros(len(colors))
  floor_color = torch.zeros(len(colors))
  annotations = sample['annotations']
  if annotations[0] != "[":
    continue
  #print(annotations)
  annotations = annotations.replace("[","").replace("]","")
  #print(annotations)
  c = annotations.split("\n")
  if len(c) >= 7:
    continue
  for i in c:
    if len(i) <= 4:
      c.remove(i)

  c = c[:4]
  obj_color,obj_shape,wall_color,floor_color = c
  obj_color = obj_color.split(":")[1].strip()
  obj_shape = obj_shape.split(":")[1].strip()
  wall_color = wall_color.split(":")[1].strip()
  floor_color = floor_color.split(":")[1].strip()

  col = torch.eye(len(colors))
  shp = torch.eye(len(shapes))

  obj_color = colors.index(obj_color)
  obj_shape = shapes.index(obj_shape)
  wall_color = colors.index(wall_color)
  floor_color = colors.index(floor_color)

  obj_color = col[obj_color]
  obj_shape = shp[obj_shape]
  wall_color = col[wall_color]
  floor_color = col[floor_color]

  sample['annotations'] = torch.cat([obj_color,obj_shape,wall_color,floor_color])
  new_ds.append(sample)

# Create a txt file with the concepts
with open("C:/Users/debryu/Desktop/VS_CODE/HOME/ML/work/askLLAMA/data/llama_ds/concepts.txt", "w") as f:
  for c in colors:
    f.write("Object " + c + "\n")
  
  for s in shapes:
    f.write("Shape " + s + "\n")
  
  for c in colors:
    f.write("Wall " + c + "\n")
  
  for c in colors:
    f.write("Floor " + c + "\n")
  
  


pickle.dump(new_ds, open(new_ds_file, "wb"))


  

