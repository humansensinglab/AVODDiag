import torch
from transformers import AutoModelForCausalLM
# from typing import Optional

from tqdm.auto import tqdm
import os
from datetime import datetime as dt
from multiprocessing import Process, Queue, Manager, current_process, set_start_method

from PIL import Image
import pandas as pd
import ast
from typing import Optional, Union



class Moondream2:
    def __init__(self, gpu_ids: list[int] = None):
        self.gpu_ids = gpu_ids # if GPU ids are not provided, the model will run on CPU

        if self.gpu_ids is None or len(self.gpu_ids) == 0:
            print('No GPU IDs provided, running on CPU only.')
            self.device_names = ["cpu"]

        elif len(self.gpu_ids) == 1:
            print(f"Single GPU ID provided, running on GPU {self.gpu_ids[0]}.")
            self.device_names = [f"cuda:{self.gpu_ids[0]}"]

        else:
            print(f"Multiple GPU IDs provided, running on GPUs {self.gpu_ids}.")
            self.device_names = [f"cuda:{gpu_id}" for gpu_id in self.gpu_ids]

    def list_models(self):
        models = [
            dict(name='vikhyatk/moondream2', display_name='Moondream 2', revision='2025-06-21'),
        ]

        return pd.DataFrame(models)


    def __init_model__(
        self,
        model_name: str,
        model_revision: str,
        device_name: str
    ):
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            revision=model_revision,
            trust_remote_code=True,
            device_map={"": device_name}
        )

        return model
    

    # Worker process that gets image paths from the queue and performs inference
    def worker(self, device_name, model_name, model_revision, f, task_queue, output_list, progress_queue):
        print(f"[Process {current_process().name}] Using {device_name}.")
        torch.cuda.set_device(device_name)

        # Init model
        model = self.__init_model__(
            model_name=model_name,
            model_revision=model_revision,
            device_name=device_name
        )
        
        # model.model.compile()
        # model.compile()

        print(f"[Process {current_process().name}] Model loaded on {device_name}. Awaiting tasks...")

        while True:
            if task_queue.empty():
                break

            try:
                task_args = task_queue.get(timeout=5)

            except:
                break

            try:
                task_args['model'] = model
                image_file_path = task_args.pop('image_file_path')
                output = f(image_file_path, **task_args)
                output_list.append(output)
                progress_queue.put(1)  # Report progress

            except Exception as e:
                print(f"[{current_process().name}] Failed to process {task_args}: {e}")


    def __init_multiprocessing__(
        self,
        kwargs_images: pd.DataFrame,
        # image_file_paths: list, 
        function_to_call,
        model_name: str,
        model_revision: str,
        **kwargs: dict
    ) -> list:
        """
        This functions perform multi-processing task execution. 
        The task is represented by the function_to_call argument, 
        which is executed on each image file path in the image_file_paths list. 
        The function is executed in parallel across multiple GPUs specified in the gpu_ids attribute of the class instance.

        Args:
            image_file_paths (list): List of image file paths to process.
            function_to_call (callable): Function to be called for each image file path.
            model_name (str): Name of the model to be used for processing.
            model_revision (str): Revision of the model to be used for processing.
            **kwargs (dict): Additional keyword arguments to be passed to the function_to_call.
        """

        task_queue = Queue()
        manager = Manager()
        output_list = manager.list()
        progress_queue = manager.Queue()

        # Populate the shared task queue
        for _, kwargs_image in kwargs_images.iterrows():
            task_queue.put(
                {
                    **kwargs_image.to_dict(),
                    **kwargs
                }
            )

        processes = []
        # for i in self.gpu_ids:
        for device_name in self.device_names:
            p = Process(
                target=self.worker, 
                args=(
                    device_name, 
                    model_name, 
                    model_revision,
                    function_to_call,
                    task_queue, 
                    output_list, 
                    progress_queue
                ), 
                name=f"GPU-{device_name}")
            
            p.start()
            processes.append(p)

        total_tasks = len(kwargs_images)
        with tqdm(total=total_tasks, desc="Processing Images") as pbar:
            completed = 0
            while completed < total_tasks:
                progress_queue.get()
                completed += 1
                pbar.update(1)

        for p in processes:
            p.join()

        return list(output_list)


    def caption_image(
            self, 
            image: Union[str, Image.Image],
            model_name:str=None,
            **kwargs: dict
    ) -> dict:
        
        if isinstance(image, str):
            image_file_path = image
            image_file_name = os.path.basename(image_file_path)
            image = Image.open(image_file_path)

        elif isinstance(image, Image.Image):
            image_file_path = None
            image_file_name = None

        else:
            raise ValueError("Invalid image type. Must be a file path string or a PIL Image object.")

        caption_length = kwargs.get('caption_length', 'normal')

        model = kwargs.get('model', None)
        if model is None:
            assert model_name is not None, "Model name must be specified in the keyword arguments."
            model_revision = kwargs.get('model_revision', None)
            device_name = kwargs.get('device_name', self.device_names[0])

            model = self.__init_model__(
                model_name=model_name,
                model_revision=model_revision,
                device_name=device_name
            )

        response = model.caption(
            image=image,
            length=caption_length
        )

        return {
            'image_file_name': image_file_name,
            f'caption:{caption_length}': response['caption']
        }


    def caption_images(
        self,
        image_file_paths: list,
        model_name: str,
        **kwargs: dict
    ) -> pd.DataFrame:

        model_revision = kwargs.get('model_revision', None)
        caption_length = kwargs.get('caption_length', 'normal')

        captions_pd = pd.DataFrame(
            self.__init_multiprocessing__(
                image_file_paths=image_file_paths,
                function_to_call=self.caption_image,
                model_name=model_name,
                model_revision=model_revision,
                caption_length=caption_length
            )
        ).sort_values(by='image_file_name').reset_index(drop=True)

        return captions_pd


    def detect_objects_image_bbox2d(
        self,
        image: Union[str, Image.Image],
        object_name: str,
        model_name:str=None,
        **kwargs: dict
    ) -> dict:

        if isinstance(image, str):
            image_file_path = image
            image_file_name = os.path.basename(image_file_path)
            image = Image.open(image_file_path)
        elif isinstance(image, Image.Image):
            image_file_path = None
            image_file_name = None
        else:
            raise ValueError("Invalid image type. Must be a file path string or a PIL Image object.")

        model = kwargs.get('model', None)
        if model is None:
            assert model_name is not None, "When model is not provided a model name must be specified in the keyword arguments."
            model_revision = kwargs.get('model_revision', None)
            device_name = kwargs.get('device_name', self.device_names[0])

            model = self.__init_model__(
                model_name=model_name,
                model_revision=model_revision,
                device_name=device_name
            )

        response = model.detect(
            image=image,
            object=object_name,
        )

        # Convert the bounding boxes to a common format: XYHW in pixel space (COCO format).
        bboxes_xywh = []
        for bbox_ltrb_norm in response['objects']:
            xmin_norm, ymin_norm, xmax_norm, ymax_norm = [bbox_ltrb_norm[k] for k in ('x_min', 'y_min', 'x_max', 'y_max')]
            x_min = xmin_norm * image.width
            y_min = ymin_norm * image.height
            x_max = xmax_norm * image.width
            y_max = ymax_norm * image.height

            w = x_max - x_min
            h = y_max - y_min
            bbox_xywh = (x_min, y_min, w, h)

            bboxes_xywh.append(bbox_xywh)

        return {
            'image_file_name': image_file_name,
            'bbox2d_norm_raw': response['objects'],
            'annotations:bbox2d_xywh_px': bboxes_xywh
        }


    def detect_objects_images_bbox2d(
        self,
        image_file_paths: list,
        object_name: str,
        model_name: str,
        **kwargs: dict
    ) -> pd.DataFrame:

        model_revision = kwargs.get('model_revision', None)

        objects_data_pd = pd.DataFrame(
            self.__init_multiprocessing__(
                kwargs_images=pd.DataFrame([{'image_file_path': image_file_path} for image_file_path in image_file_paths]),
                function_to_call=self.detect_objects_image_bbox2d,
                model_name=model_name,
                model_revision=model_revision,
                object_name=object_name
            )
        ).sort_values(by='image_file_name').reset_index(drop=True)

        return objects_data_pd


    def query_image_prompt(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        model_name:str=None,
        **kwargs: dict
    ):

        parse_response = kwargs.get('is_json_response', False)
        
        if isinstance(image, str):
            image_file_path = image
            image_file_name = os.path.basename(image_file_path)
            image = Image.open(image_file_path)
        elif isinstance(image, Image.Image):
            image_file_path = None
            image_file_name = None
        else:
            raise ValueError("Invalid image type. Must be a file path string or a PIL Image object.")

        model = kwargs.get('model', None)
        if model is None:
            assert model_name is not None, "Model name must be specified in the keyword arguments."
            model_revision = kwargs.get('model_revision', None)
            device_name = kwargs.get('device_name', self.device_names[0])

            model = self.__init_model__(
                model_name=model_name,
                model_revision=model_revision,
                device_name=device_name
            )

        response = model.query(
            image=image,
            question=prompt,
        )

        output = {
            'image_file_name': image_file_name,
            'response_raw': response['answer'],
        }

        if parse_response:
            try:
                output['response_parsed'] = ast.literal_eval(response['answer'])
                
            except:
                output['response_parsed'] = None

        return output


    def query_images_prompt(
        self,
        image_file_paths: list,
        prompt: str,
        model_name: str,
        **kwargs: dict
    ) -> pd.DataFrame:
        assert False, "This function is deprecated. Use query_images_prompts instead."
        print('Num. Images: ', len(image_file_paths))
        
        model_revision = kwargs.get('model_revision', None)
        parse_response = kwargs.get('is_json_response', False)
        
        responses_pd = pd.DataFrame(
            self.__init_multiprocessing__(
                image_file_paths=image_file_paths,
                function_to_call=self.query_image_prompt,
                model_name=model_name,
                model_revision=model_revision,
                prompt=prompt,
                is_json_response=parse_response
            )
        ).sort_values(by='image_file_name').reset_index(drop=True)

        return responses_pd


    def query_images_prompts(
        self,
        query_metadata: pd.DataFrame,
        model_name: str,
        **kwargs: dict
    ) -> pd.DataFrame:

        print('Num. Images: ', len(query_metadata))

        model_revision = kwargs.get('model_revision', None)
        parse_response = kwargs.get('is_json_response', False)
        
        responses_pd = pd.DataFrame(
            self.__init_multiprocessing__(
                kwargs_images=query_metadata,
                function_to_call=self.query_image_prompt,
                model_name=model_name,
                model_revision=model_revision,
                is_json_response=parse_response
            )
        ).sort_values(by='image_file_name').reset_index(drop=True)

        return responses_pd