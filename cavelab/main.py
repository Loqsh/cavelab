from cavelab import h5data

path = '/FilterFinder/data/prealigned/'
reader = h5data(path)


print(reader.read('6,7_prealigned', (1000, 1000), 512))
