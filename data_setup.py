import bdlb
from bdlb.fishyscapes.benchmark_road import FishyscapesOnRoad_RODataset

bench = FishyscapesOnRoad_RODataset()
bench.get_dataset()
for blob in bench.get_dataset():
    print(blob.keys())
    break

