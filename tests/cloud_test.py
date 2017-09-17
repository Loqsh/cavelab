from cavelab import Cloud, visual
pinky = Cloud('gs://neuroglancer/pinky40_v11/image', mip=2)
image = pinky.read((59476, 11570, 714), 100, 100)
print(image.shape)
