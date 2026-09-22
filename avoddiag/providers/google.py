import base64

from google import genai
from google.genai import types
from PIL import Image
import io
import os
import pandas as p
from datetime import datetime as dt
import pandas as pd
from typing import Optional, Union

from tqdm.auto import tqdm
import json
import textwrap


def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    # json_found = False
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            # json_found = True
            break  # Exit the loop once "```json" is found

    return json.loads(json_output)
        

class GoogleGenAI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = genai.Client(api_key=self.api_key)

    def list_models(self):
        """
        Print the names of all available models that the provider can use.

        Returns:
            A Pandas table with all models information.
        """

        models = []
        for model in self.client.models.list():
            assert model.name.find('models/') == 0

            models.append(
                dict(
                    name=model.name.replace('models/', ''),
                    display_name=model.display_name,
                    description=model.description,
                    input_token_limit=model.input_token_limit,
                    output_token_limit=model.output_token_limit,
                    supported_actions=model.supported_actions,
                    tuned_model_info=model.tuned_model_info,
                    version=model.version
                )
            )

        return pd.DataFrame(models)
    

    def generate_image(
            self,
            prompt: str,
            model_name: str,
            **kwargs: dict
    ) -> dict:
        num_attempts = kwargs.get('num_attempts', 3)

        model_info = self.client.models.get(model=model_name)
        
        pil_image = None
        elapsed_time = None
        e = None
        success = False
        for i_attempt in range(num_attempts):
            success = False
            
            if 'generateContent' in model_info.supported_actions:
                try:
                    t_start = dt.now()
                    # response = self.client.models.generate_content(
                    #     model=model_name,
                    #     contents=[prompt],
                    # )
                    interaction = self.client.interactions.create(
                        model=model_name,
                        input=prompt,
                        response_format={
                            "type": "image",
                            "mime_type": "image/jpeg",
                            "aspect_ratio": "1:1",
                            "image_size": "1K"
                        },
                    )
                    t_end = dt.now()
                    elapsed_time = t_end - t_start

                    # filtered_parts = list(filter(lambda p: p.inline_data is not None, response.candidates[0].content.parts))
                    # assert len(filtered_parts) == 1, "Only one inline data part is expected."

                    # pil_image = Image.open(io.BytesIO(filtered_parts[0].inline_data.data))
                    pil_image = Image.open(io.BytesIO(base64.b64decode(interaction.output_image.data)))
                    success = True
                
                except Exception as e:
                    print(f"Error generating image with 'generateContent' for model '{model_name}': {e}")
                    continue

            elif 'predict' in model_info.supported_actions:
                try:
                    t_start = dt.now()
                    response = self.client.models.generate_images(
                        model=model_name,
                        prompt=prompt,
                        config=types.GenerateImagesConfig(
                            number_of_images=1,
                        )
                    )
                    t_end = dt.now()
                    elapsed_time = t_end - t_start

                    assert len(response.generated_images) == 1, f"Expected exactly 1 generated image, but got {len(response.generated_images)}."

                    pil_image = Image.open(io.BytesIO(response.generated_images[0].image.image_bytes))
                    success = True
                
                except Exception as e:
                    print(f"Error generating image with 'predict' for model '{model_name}': {e}")
                    continue
            
            else:
                raise ValueError(f"The model '{model_name}' does not support image generation actions.")

            if success:
                break

        return {
            'image': pil_image,
            'generation_status': 'Success' if success else 'Failed',
            'exception': None if success else f'{e}',
            'elapsed_time_sec': elapsed_time.total_seconds(),
        }
    

    def generate_images(
        self,
        prompts: list,
        model_name: str,
        images_folder_path: str,
        **kwargs: dict
    ):
        # os.makedirs(images_folder_path, exist_ok=False)

        images_generated_data = []
        for i_prompt, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Generating images"):
            image_data = self.generate_image(
                prompt=prompt,
                model_name=model_name,
                **kwargs
            )

            image_file_name = f'{i_prompt:06d}.png'
            image_file_path = os.path.join(images_folder_path, image_file_name)
            image_data['image'].save(image_file_path)
            # print(f'Saved image: {image_file_path}')

            images_generated_data.append({
                'image_file_name': image_file_name,
                'prompt': prompt,
                'generation_status': image_data['generation_status'],
                'exception': image_data['exception'],
                'elapsed_time_sec': image_data['elapsed_time_sec']
            })

        return pd.DataFrame(images_generated_data).sort_values(by='image_file_name').reset_index(drop=True)



    def query_image_prompt(
            self,
            image: Union[str, Image.Image],
            prompt: str,
            model_name: str,
            **kwargs: dict
    ) -> dict:
        """
        Args:
            image: A file path to the image or a PIL Image object.
            prompt: The text prompt to query the image with.
            model_name: The name of the model to use for querying.
            kwargs: Additional keyword arguments for the model query.
        """
        
        config = kwargs.get('config', None)
        parse_response = kwargs.get('is_json_response', False)

        if config is not None and parse_response:
            config.response_mime_type = 'application/json'

        if isinstance(image, str):
            image_file_path = image
            image_file_name = os.path.basename(image_file_path)
            image = Image.open(image_file_path)

        elif isinstance(image, Image.Image):
            image_file_path = None
            image_file_name = None
            
        else:
            raise ValueError("Invalid image type. Must be a file path string or a PIL Image object.")

        output = {
            'image_file_name': image_file_name,
            'response_raw': None,
            'query_status': None,
            'exception': None
        }

        response = None
        elapsed_time = None
        success = False
        e = None
        try:
            t_start = dt.now()
            response = self.client.models.generate_content(
                model=model_name,
                contents=[
                    image,
                    prompt
                ],
                config=config
            )
            t_end = dt.now()
            elapsed_time = t_end - t_start
            success = True

        except Exception as e:
            # raise RuntimeError(f"Error querying image and prompt with model '{model_name}': {e}")
            print(f"Error querying image ({image_file_name}) and prompt with model '{model_name}': {e}")
            output['exception'] = f'{e}'

        output['query_status'] = 'Success' if success else 'Failed'
        output['elapsed_time_sec'] = elapsed_time.total_seconds() if elapsed_time is not None else None
        output['response_raw'] = response.text if response is not None else None

        if parse_response:
            response_parsed = None
            response_parsing_success = False
            if success and (response is not None) and (len(response.text) > 0):
                try:
                    response_parsed = parse_json(response.text)
                    response_parsing_success = True

                except Exception as e:
                    # print(f"Error parsing JSON response for image ({image_file_name}) and prompt '{prompt}' with model '{model_name}': '{e}', response text: '{response.text}'")
                    print(f"Error parsing JSON response for image ({image_file_name}) and prompt '{prompt}' with model '{model_name}': '{e}'")
                    response_parsed = None
                    response_parsing_success = False

            # else:
            #     response_parsed = 'Unsuccessful'

            output['response_parsed'] = response_parsed
            output['is_parsing_successful'] = response_parsing_success 

        return output


    def query_images_prompt(
        self,
        image_file_paths: list,
        prompt: Union[str, list[str]],
        model_name: str,
        **kwargs: dict
    ) -> pd.DataFrame:

        """
        This provider currently only supports caching responses to files in order to minimize potential data losses due to connection disturbances or server overloading.
        Hence, the 'responses_cache_folder_path' argument must be provided in 'kwargs'.
        """

        # Cache responses to files
        responses_cache_folder_path = kwargs.get('responses_cache_folder_path', None)
        assert responses_cache_folder_path is not None, "This provider does not support in-memory responses caching. 'responses_cache_folder_path' must be provided."
        os.makedirs(responses_cache_folder_path, exist_ok=True)

        # Process requests and cache responses
        if isinstance(prompt, str):
            prompt = [prompt] * len(image_file_paths)

        assert len(prompt) == len(image_file_paths), "Length of prompt list must match the number of image file paths."

        for image_file_path, image_prompt in tqdm(zip(image_file_paths, prompt), total=len(image_file_paths), desc="Querying images and prompt"):
            response_file_path = os.path.join(
                responses_cache_folder_path,
                f"{os.path.splitext(os.path.basename(image_file_path))[0]}.json"
            )
            if os.path.isfile(response_file_path):
                print(f"Response for image '{image_file_path}' already exists. Skipping query.")
                continue

            response = self.query_image_prompt(
                image=image_file_path,
                prompt=image_prompt,
                model_name=model_name,
                **kwargs
            )

            with open(response_file_path, "w") as f:
                json.dump(response, f)


        # Load all cached responses
        images_generated_data = []
        for image_file_path in tqdm(image_file_paths, desc="Loading cached responses"):
            response_file_path = os.path.join(
                responses_cache_folder_path,
                f"{os.path.splitext(os.path.basename(image_file_path))[0]}.json"
            )
            with open(response_file_path, "r") as f:
                response = json.load(f)

            images_generated_data.append(response)

        return pd.DataFrame(images_generated_data).sort_values(by='image_file_name').reset_index(drop=True)


    def query_images_prompts(
        self,
        query_metadata: pd.DataFrame,
        model_name: str,
        **kwargs: dict
    ) -> pd.DataFrame:
        # Cache responses to files
        responses_cache_folder_path = kwargs.get('responses_cache_folder_path', None)
        assert responses_cache_folder_path is not None, "This provider does not support in-memory responses caching. 'responses_cache_folder_path' must be provided."
        os.makedirs(responses_cache_folder_path, exist_ok=True)

        for i_row, row in tqdm(query_metadata.iterrows(), total=len(query_metadata), desc="Querying images and prompts"):
            image_file_path = row['image_file_path']
            image_prompt = row['prompt']

            response_file_path = os.path.join(
                responses_cache_folder_path,
                f"{os.path.splitext(os.path.basename(image_file_path))[0]}.json"
            )
            if os.path.isfile(response_file_path):
                print(f"Response for image '{image_file_path}' already exists. Skipping query.")
                continue

            response = self.query_image_prompt(
                image=image_file_path,
                prompt=image_prompt,
                model_name=model_name,
                **kwargs
            )

            with open(response_file_path, "w") as f:
                json.dump(response, f)


        # Load all cached responses
        images_generated_data = []
        for image_file_name in tqdm(query_metadata['image_file_name'], desc="Loading cached responses"):
            response_file_path = os.path.join(
                responses_cache_folder_path,
                f"{os.path.splitext(image_file_name)[0]}.json"
            )
            with open(response_file_path, "r") as f:
                response = json.load(f)

            images_generated_data.append(response)

        return pd.DataFrame(images_generated_data).sort_values(by='image_file_name').reset_index(drop=True)


    def detect_objects_image_bbox2d(
        self,
        # image: Union[str, Image.Image],
        image_file_path: str,
        object_name: str,
        model_name:str,
        **kwargs: dict
    ) -> dict:
        config = kwargs.get('config', None)
        # parse_response = kwargs.get('is_json_response', False)
        
        # if isinstance(image, str):
            # image_file_path = image
        image_file_name = os.path.basename(image_file_path)
        image = Image.open(image_file_path)
        image_size = image.size  # (width, height)
        image.close()

        # elif isinstance(image, Image.Image):
        #     image_file_path = None
        #     image_file_name = None

        # else:
        #     raise ValueError("Invalid image type. Must be a file path string or a PIL Image object.")
    
        bounding_box_system_instructions = textwrap.dedent(
            "Return bounding boxes as a JSON array with labels. Never return masks or code fencing."
        )

        prompt = textwrap.dedent(
            f"""Detect all {object_name} in the image. 
            The box_2d should be [xmin, ymin, xmax, ymax] normalized to 0-1000. 
            Use descriptive labels."""
        )

        if config is None:
            config = types.GenerateContentConfig(
                system_instruction=bounding_box_system_instructions,
                temperature=0.0,

                thinking_config=types.ThinkingConfig(
                    thinking_budget=0
                ),
                response_mime_type = 'application/json'
            )

        if config is not None and config.response_mime_type is None:
            config.response_mime_type = 'application/json'

        response = self.query_image_prompt(
            # image=image,
            image=image_file_path,
            prompt=prompt,
            model_name=model_name,
            config=config,
            is_json_response=True
        )
        response['image_file_name'] = image_file_name

        # If there are no detected objects, ensure the parsed response is an empty list
        if (response['response_parsed'] is None) or (not isinstance(response['response_parsed'], list)):
            response['response_parsed'] = []

        response['annotations:bbox2d_xywh_px'] = self.bbox2d_denormalize(response['response_parsed'], image_size)

        return response


    def bbox2d_denormalize(self, response_parsed, image_size):
        bboxes_xywh = []
        for anno in response_parsed:
            xmin = image_size[0]*anno['box_2d'][1]/1000
            ymin = image_size[1]*anno['box_2d'][0]/1000
            xmax = image_size[0]*anno['box_2d'][3]/1000
            ymax = image_size[1]*anno['box_2d'][2]/1000

            w = xmax - xmin
            h = ymax - ymin

            bboxes_xywh.append([xmin, ymin, w, h])

        return bboxes_xywh


    def detect_objects_images_bbox2d(
        self,
        image_file_paths: list,
        object_name: str,
        model_name: str,
        **kwargs: dict
    ) -> pd.DataFrame:

        # Cache responses to files
        responses_cache_folder_path = kwargs.get('responses_cache_folder_path', None)
        assert responses_cache_folder_path is not None, "This provider does not support in-memory responses caching. 'responses_cache_folder_path' must be provided."
        os.makedirs(responses_cache_folder_path, exist_ok=True)
        
        # Process requests and cache responses
        for image_file_path in tqdm(image_file_paths, desc="Processing queries for image object detection"):
            response_file_path = os.path.join(
                responses_cache_folder_path,
                f"{os.path.splitext(os.path.basename(image_file_path))[0]}.json"
            )
            if os.path.isfile(response_file_path):
                print(f"Response for image '{image_file_path}' already exists. Skipping query.")
                continue
            
            response = self.detect_objects_image_bbox2d(
                # image=image_file_path,
                image_file_path=image_file_path,
                object_name=object_name,
                model_name=model_name
            )

            with open(response_file_path, "w") as f:
                json.dump(response, f)

        # Load all cached responses
        images_detection_data = []
        for image_file_path in tqdm(image_file_paths, desc="Loading cached responses"):
            response_file_path = os.path.join(
                responses_cache_folder_path,
                f"{os.path.splitext(os.path.basename(image_file_path))[0]}.json"
            )
            with open(response_file_path, "r") as f:
                response = json.load(f)

            images_detection_data.append(response)

        return pd.DataFrame(images_detection_data).sort_values(by='image_file_name').reset_index(drop=True)
