import os
from pickle import load
import pandas as pd
from PIL import Image

from tqdm.auto import tqdm

from typing import Union


class Dataset():
    """
    A class to represent image datasets.
    """

    def __init__(
            self, 
            root_folder_path: str,
        ):

        """
        Initializes a new ImageDataset object.

        Args:
            root_folder_path: The root folder path containing the dataset.
            metadata_keys: represent different types of metadata associated with each image.
            load_data: Whether to load image and metadata information. For new/empty datasets, it should be left False.
        """

        self.root_folder_path = root_folder_path
        self.images_folder_path = os.path.join(root_folder_path, "images")
        self.metadata_folder_path = os.path.join(root_folder_path, "metadata")
        self.cache_folder_path = os.path.join(root_folder_path, "cache")

        self.image_file_names = []
        self.image_types_allowed = {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}

        self.metadata = dict()

        if os.path.isdir(self.root_folder_path):
            print(f'Loading dataset from root folder path: "{self.root_folder_path}"')
            self.load()


    def save(self, overwrite=False):
        if not overwrite and os.path.isfile(self.input_attrib_metadata_file_path):
            raise FileExistsError(f"Input attribute metadata file {self.input_attrib_metadata_file_path} already exists.")

        if not overwrite and os.path.isfile(self.generated_attrib_metadata_file_path):
            raise FileExistsError(f"Generated attribute metadata file {self.generated_attrib_metadata_file_path} already exists.")

        if not overwrite and os.path.isfile(self.annotations_file_path):
            raise FileExistsError(f"Annotations file {self.annotations_file_path} already exists.")

        self.input_attrib_metadata.to_json(self.input_attrib_metadata_file_path)
        self.generated_attrib_metadata.to_json(self.generated_attrib_metadata_file_path)
        self.annotations.to_json(self.annotations_file_path)


    def make_dirs(self):
        os.makedirs(self.root_folder_path, exist_ok=False)
        os.makedirs(self.images_folder_path, exist_ok=False)
        os.makedirs(self.metadata_folder_path, exist_ok=False)


    def load(self):
        self._load_image_file_names_()
        self._load_metadata_()


    def _load_image_file_names_(self):
        print(f'\nLoading images data from  folder "{os.path.relpath(self.images_folder_path, self.root_folder_path)}/"...')
        # List image files
        assert os.path.isdir(self.images_folder_path), 'Images folder does not exist. Unable to load image files.'

        self.image_file_names = sorted(
            filter(
                lambda fn: os.path.splitext(fn)[1] in self.image_types_allowed,
                os.listdir(self.images_folder_path)
            )
        )

        print(f'   Number of image files found: {len(self.image_file_names):,}')


    def _load_metadata_(self):
        """
        Load metadata stored on the hard drives. All metadata files are Pandas tables.
        """

        print('\nLoading metadata...')
        metadata_file_paths = {}
        for root, _, files in os.walk(self.metadata_folder_path):
            if len(files) == 0:
                continue

            json_file_names = sorted(filter(lambda fn: fn.endswith('.json'), files))

            if len(json_file_names) == 0:
                continue

            metadata_folder_relpath_split = os.path.relpath(root, self.metadata_folder_path).split(os.sep)
            if metadata_folder_relpath_split and metadata_folder_relpath_split[0] == ".":
                metadata_folder_relpath_split = metadata_folder_relpath_split[1:]

            for fn in json_file_names:
                fp = os.path.join(root, fn)
                metadata_key = ':'.join(metadata_folder_relpath_split + [os.path.splitext(fn)[0]])
                metadata_file_paths[metadata_key] = fp

        image_file_names_set = set(self.image_file_names)
        num_images = len(self.image_file_names)
        metadata_key_len_max = max([len(k) for k in metadata_file_paths.keys()]) if metadata_file_paths else 0
        for metadata_key, fp in metadata_file_paths.items():
            metadata_key_str = f'Key "{metadata_key}"'
            print(f'   {metadata_key_str: <{metadata_key_len_max+6}}', end=' ')
            metadata = pd.read_json(fp)
            assert 'image_file_name' in metadata.columns, f'Metadata file "{fp}" does not contain required "image_file_name" column.'
            assert metadata['image_file_name'].is_unique, f'Metadata file "{fp}" contains non-unique values in "image_file_name" column.'
            assert set(metadata['image_file_name']).issubset(image_file_names_set), f'Metadata file "{fp}" contains image file names not present in the dataset.'
            
            self.metadata[metadata_key] = metadata
            print(f' | Num. rows: {metadata.shape[0]:,d}', end='')
            if metadata.shape[0] != num_images:
                print(f' (WARNING: Number of rows does not match number of images!)', end='')
            print(f' | Loaded from "{os.path.relpath(fp, self.root_folder_path)}".')

        if not self.metadata:
            print('   No metadata files found.')



    def __len__(self):
        return len(self.image_file_names)


    def __getitem__(
            self, 
            id: Union[int, str, slice, list, tuple]
        ) -> dict:
        """
        Get an image and its associated metadata by index or image file name.

        Args:
            id: An integer index or a string representing the image file name.
        """
        if isinstance(id, int):
            image_file_name = self.image_file_names[id]

        elif isinstance(id, str):
            image_file_name = id
            assert image_file_name in self.image_file_names, f"Image file name '{image_file_name}' not found in dataset."

        elif isinstance(id, (slice, list, tuple)):
            return [self[i] for i in range(*id.indices(len(self)))] if isinstance(id, slice) else [self[i] for i in id]

        else:
            raise ValueError("Invalid id type. Must be int or str.")

        image_file_path = os.path.join(self.images_folder_path, image_file_name)
        image = Image.open(image_file_path)

        image_metadata = {}
        for metadata_key, metadata in self.metadata.items():
            d = metadata[metadata['image_file_name'] == image_file_name]
            if d.shape[0] == 0:
                continue
            
            image_metadata[metadata_key] = d.iloc[0]

        return {
            "image_file_name": image_file_name,
            "image_file_path": image_file_path,
            "image": image,
            "metadata": image_metadata
        }


    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


    def save_metadata(self, overwrite_all_files: bool = False, keys_to_overwrite:list = []):
        for metadata_key, metadata in self.metadata.items():
            if (metadata is None) or (metadata.shape[0] == 0):
                continue

            # metadata_file_path = os.path.join(self.metadata_folder_path, f"{metadata_key}.json")
            metadata_file_path = os.path.join(self.metadata_folder_path, *metadata_key.split(':')) + '.json'

            if os.path.isfile(metadata_file_path) and (not overwrite_all_files) and (metadata_key not in keys_to_overwrite):
                print(f"Metadata file {metadata_file_path} already exists. Skipping.")
                continue

            print(f'Saving metadata file "{metadata_file_path}"', end='')
            os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)
            metadata.to_json(metadata_file_path)
            print(f' | Saved {metadata.shape[0]:,} rows and {metadata.shape[1]:,} columns.')


    def __repr__(self):
        return f"Dataset(root_folder_path='{self.root_folder_path}', num_images={len(self)})"

