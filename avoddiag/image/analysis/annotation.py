import os
import json
import pandas as pd
from tqdm.auto import tqdm

from ..data import Dataset

"""
Here different annotation types can be mapped to different provider methods. 
E.g., 'bbox2d' to 'detect_objects_images_bbox2d' or 'segmentation' to 'segment_objects_images'.
"""
annotation_hooks = {
    'bbox2d': 'detect_objects_images_bbox2d',
}


class AnnotationExtractor:
    def __init__(
        self, 
        provider, 
        model_name, 
        with_caching: bool=False,
        type: str='bbox2d',
        **kwargs: dict
    ):
        self.provider = provider
        self.model_name = model_name
        self.with_caching = with_caching
        self.type = type
        self.kwargs = kwargs

        assert self.model_name.find(':') == -1, "Invalid model name format. ':' is an illegal character."
        assert self.type in ['bbox2d'], "Unsupported annotation type. Supported types: 'bbox2d'."


    def extract_dataset_annotations(
        self,
        image_dataset: Dataset,
        object_name: str,
        image_file_paths_to_process: list=None,
    ):
        kwargs = self.kwargs.copy()

        assert len(image_dataset) > 0, "Dataset is empty. No annotations to extract."

        assert object_name.find(':') == -1, "Invalid object name format. ':' is an illegal character."
        metadata_key = ':'.join(
            [
                'annotations_generated',
                f'{self.model_name.replace("/", "+")}',
                f'{object_name.replace("/", "+")}'
            ]
        )
        assert metadata_key not in image_dataset.metadata, f"Metadata already contains generated annotations for model '{self.model_name}'"

        if self.with_caching:
            print('Responses caching is enabled. Queries with existing responses will be skipped.')
            responses_cache_folder_path = os.path.join(
                image_dataset.cache_folder_path,
                *metadata_key.split(':')
            )
            print(f'Cache folder path: {responses_cache_folder_path}')
            # os.makedirs(responses_cache_folder_path, exist_ok=True) # The folder will be created by the provider

            kwargs['responses_cache_folder_path'] = responses_cache_folder_path
        
        print('Initiate annotation extraction process')

        f_detect_objects = getattr(self.provider, annotation_hooks[self.type])

        if image_file_paths_to_process is None:
            image_file_paths_to_process = [image_data['image_file_path'] for image_data in image_dataset]

        annotations_generated = f_detect_objects(
            image_file_paths=image_file_paths_to_process,
            object_name=object_name,
            model_name=self.model_name,
            **kwargs
        )
        
        image_dataset.metadata[metadata_key] = annotations_generated
        image_dataset.save_metadata(keys_to_overwrite=[metadata_key])
        print('Finished saving metadata.')

        # return annotations_generated