from pathlib import Path
import argparse
from domainbed.datasets import datasets

from ffcv.fields import IntField, RGBImageField
from ffcv.writer import DatasetWriter


def write(args):
    
    dataset = vars(datasets)[args.dataset](args.data_dir)
    print(f'Writting beton files for {args.dataset} dataset')

    for env, ds in zip(dataset.environments, dataset.datasets):
        print(f'Writting {env} beton file')
        writer = DatasetWriter(Path(ds.root)/f'{env}.beton', {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)   

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ffcv writer")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/projets/masih/dataset")
    args = parser.parse_args()

    write(args)