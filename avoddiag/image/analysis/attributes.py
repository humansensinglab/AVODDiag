
import os
import json
import pandas as pd
from tqdm.auto import tqdm
from typing import Union
from PIL import Image

from ..data import Dataset



class AttributeExtractor:
    def __init__(
        self, 
        provider, 
        model_name, 
        with_caching: bool=False,
        **kwargs: dict
    ):
        self.provider = provider
        self.model_name = model_name
        self.with_caching = with_caching
        self.kwargs = kwargs


    def extract_image_attributes(
        self,
        image: Union[str, Image.Image],
        prompt: str,
    ) -> dict:
        kwargs = self.kwargs.copy()
        kwargs['is_json_response'] = True  # Ensure the response is parsed as JSON

        if isinstance(image, Image.Image):
            image_file_path = None

        elif isinstance(image, str):
            image_file_path = image
            image = Image.open(image_file_path)

        else:
            raise ValueError("Image must be a file path (str) or a PIL Image object.")
        
        attributes_generated = self.provider.query_image_prompt(
            image=image,
            prompt=prompt,
            model_name=self.model_name,
            **kwargs
        )
        
        return attributes_generated
    

    def extract_dataset_attributes(
        self,
        image_dataset: Dataset,
        query_metadata: pd.DataFrame,
        # prompt: Union[str, list[str]],
        metadata_name: str = None
    ):
        """
        Args:
            image_dataset (Dataset): An instance of the Dataset class containing images and existing metadata.
            query_metadata (pd.DataFrame): A pandas DataFrame containing prompts for attribute extraction. It should have a columns named 'image_file_name' and 'prompt'.
        """
        assert len(image_dataset) > 0, "Dataset is empty. No attributes to extract."

        assert 'image_file_name' in query_metadata.columns, "Query metadata must contain 'image_file_name' column."
        assert 'prompt' in query_metadata.columns, "Query metadata must contain 'prompt' column."
        assert query_metadata.shape[0] > 0, "Query metadata is empty. Prompts are needed to extract attributes."
        assert len(query_metadata['image_file_name'].unique()) == len(query_metadata), "Each image_file_name in query_metadata must be unique."
        assert query_metadata['image_file_name'].isin(image_dataset.image_file_names).all(), "Some image_file_name in query_metadata do not exist in the image_dataset."

        kwargs = self.kwargs.copy()
        kwargs['is_json_response'] = True  # Ensure the response is parsed as JSON

        metadata_key_elems = [
            'attributes_generated',
            f'{self.model_name.replace("/", "+")}'
        ]
        if metadata_name is not None:
            metadata_key_elems.append(metadata_name)

        metadata_key = ':'.join(metadata_key_elems)
        assert '/' not in metadata_key, "Metadata key cannot contain '/' character."
        assert metadata_key not in image_dataset.metadata, f"Metadata already contains generated attributes for model '{self.model_name}'"
        
        if self.with_caching:
            print('Responses caching is enabled. Queries with existing responses will be skipped.')
            responses_cache_folder_path = os.path.join(
                image_dataset.cache_folder_path,
                *metadata_key.split(':')
            )
            print(f'Cache folder path: {responses_cache_folder_path}')
            # os.makedirs(responses_cache_folder_path, exist_ok=True) # The folder will be created by the provider

            kwargs['responses_cache_folder_path'] = responses_cache_folder_path
        
        print('Initiate attribute extraction process')

        # if isinstance(prompt, str):
        #     prompt = [prompt] * len(image_dataset)

        # assert len(prompt) == len(image_dataset), "Length of prompt list must match the number of images in the dataset."

        query_metadata['image_file_path'] = query_metadata['image_file_name'].apply(
            lambda x: os.path.join(image_dataset.images_folder_path, x)
        )

        attributes_generated = self.provider.query_images_prompts(
            # image_file_paths=[image_data['image_file_path'] for image_data in image_dataset],
            # prompt=prompt,
            query_metadata=query_metadata,
            model_name=self.model_name,
            **kwargs
        )
        
        # Unpack responses (responses are expected to be dictionaries with attribute names as keys such that pd.DataFrame(responses) is possible)
        metadata = attributes_generated
        mask = metadata['response_parsed'].notna()
        d = metadata[mask][['image_file_name', 'response_parsed']].reset_index(drop=True)
        d_unpacked = pd.DataFrame(d['response_parsed'].tolist())
        d_unpacked = d_unpacked.rename(columns={cn: f'attribute:{cn}' for cn in d_unpacked.columns})
        d = d.merge(d_unpacked, left_index=True, right_index=True).drop(columns=['response_parsed'])
        attributes_generated = metadata.merge(d, on='image_file_name', how='left').sort_values('image_file_name').reset_index(drop=True)

        image_dataset.metadata[metadata_key] = attributes_generated
        image_dataset.save_metadata(keys_to_overwrite=[metadata_key])
        print('Finished saving metadata.')



class Analyzer:
    def __init__(self):
        raise NotImplementedError("Analyzer class is not implemented yet.")