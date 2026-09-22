from openai import OpenAI as openai_api
import pandas as pd
import json
# from textwrap import dedent
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from datetime import datetime as dt


class OpenAI:
    def __init__(self, api_key: str):
        self.api_key = api_key

        self.client = openai_api(
            api_key=api_key
        )

    def list_models(self):
        """
        Print the names of all available models that the provider can use.

        Returns:
            A Pandas table with all models information.
        """

        models = []
        for model in self.client.models.list():
            assert model.object == 'model'
            models.append(model.to_dict())

        return pd.DataFrame(models)
    

    def generate_prompts(
            self, 
            model_name: str, 
            instructions: str,
            prompt_attributes: pd.DataFrame
    ) -> pd.DataFrame:
        prompt_attributes = prompt_attributes.copy()

        # ===== 1) Define your structured output =====
        class PromptRow(BaseModel):
            row_index: int = Field(..., description="Index echoed from the CSV")
            prompt: str = Field(..., description="Compact, single- or two-sentence generation prompt")
            negative_prompt: str | None = Field(
                default=None, description="Optional: short list of things to avoid"
            )
            notes: str | None = Field(
                default=None, description="Optional: a short assumption/clarification"
            )

        class PromptBatch(BaseModel):
            rows: list[PromptRow]

        # Add a stable index so we can map results back
        prompt_attributes.insert(0, "row_index", prompt_attributes.index.astype(int))

        # ===== 4) Practical batching (good for throughput & cost) =====
        BATCH_SIZE = 30  # adjust for your workflow

        all_rows: list[PromptRow] = []

        start_indices = range(0, len(prompt_attributes), BATCH_SIZE)
        for start in tqdm(start_indices, total=len(start_indices), desc="Generating prompts"):
            chunk = prompt_attributes.iloc[start:start+BATCH_SIZE][
                # ["row_index", "scene_type", "weather", "season", "vehicle_count", "vehicle_colors"]
                list(prompt_attributes.columns)
            ].copy()

            # csv_snippet = chunk.to_csv(index=False)
            records = chunk.to_dict(orient="records")

            # IMPORTANT: Use `responses.parse(...)` with a Pydantic model.
            # This returns a validated object at `response.output_parsed`.
            print(dt.now(), f"Processing rows {start} to {start + len(chunk)}...")
            response = self.client.responses.parse(
                model=model_name,
                instructions=instructions,
                # input=(
                #     "Generate prompts for these rows (CSV):\n\n"
                #     f"{csv_snippet}\n\n"
                #     "Return an object that conforms exactly to the PromptBatch schema."
                # ),
                input=(
                    "The following is a JSON array of image-generation attribute records.\n\n"
                    f"{json.dumps(records, indent=2)}"
                ),
                text_format=PromptBatch,   # <= Structured Outputs via Pydantic
            )
            print(dt.now(), f"Completed rows {start} to {start + len(chunk)}.")

            parsed: PromptBatch = response.output_parsed
            all_rows.extend(parsed.rows)

        prompt_attributes = prompt_attributes.rename(columns=lambda x: f"attribute:{x}" if (x != "row_index") else x)

        prompt_attributes = prompt_attributes.merge(
            pd.DataFrame(
                list(map(lambda x: dict(row_index=x.row_index, prompt=x.prompt), all_rows)),
            ),
            on="row_index",
        ).drop(columns=["row_index"])

        return prompt_attributes
