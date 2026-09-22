
import os
import pandas as pd
from tqdm.auto import tqdm

from ..data import Dataset


class Generator:
    def __init__(self, provider, model_name):
        self.provider = provider
        self.model_name = model_name
    
    def generate_dataset(
        self,
        image_dataset: Dataset,
        metadata: pd.DataFrame,
        **kwargs
    ):
        """
        Args:
            image_dataset (Dataset): An instance of the Dataset class where images will be stored. It should be empty before generation.
            metadata (pd.DataFrame): A pandas DataFrame containing prompts for image generation. It should have a column named 'prompt'.
            kwargs: Additional keyword arguments to pass to the image generation provider.
        """
        num_attempts = kwargs.get('num_attempts', 3)

        assert len(image_dataset) == 0, "Dataset is not empty."

        assert metadata is not None, "'attributes_input' key is missing from the dataset metadata. It should contain prompts to generate images."
        assert isinstance(metadata, pd.DataFrame), "Input attributes metadata should be a pandas DataFrame."
        assert metadata.shape[0] > 0, "Metadata is empty. Prompts are needed to generate new images."
        assert 'prompt' in metadata.columns, "Metadata does not contain prompts. Prompts need to be generated first."
        
        print(f'Create dataset folders')
        image_dataset.make_dirs()

        print('Initiate image generation process')
        images_data = self.provider.generate_images(
            prompts=metadata['prompt'].tolist(),
            model_name=self.model_name,
            images_folder_path=image_dataset.images_folder_path,
            num_attempts=num_attempts
        )
        metadata = metadata.merge(
            images_data,
        )

        print(f'Finished generating images for {len(metadata):,d} rows.')

        assert 'attributes_input' not in image_dataset.metadata, "'attributes_input' key already exists in the dataset metadata. Overwriting it may lead to data loss."
        image_dataset.metadata['attributes_input'] = metadata
        image_dataset.save_metadata(keys_to_overwrite=['attributes_input'])
        print('Finished saving metadata.')

        print('Load newly generated data.')
        image_dataset.load()

