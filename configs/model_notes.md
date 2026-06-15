# Candidate model IDs

Use these exact model names when running baseline tests.

| Short name | Hugging Face model ID | First use |
|---|---|---|
| GPT-2 small | openai-community/gpt2 | First baseline test |
| BLOOM-560m | bigscience/bloom-560m | Second baseline test |
| SmolLM2-360M | HuggingFaceTB/SmolLM2-360M | Third baseline test |

Important:
- Start with GPT-2 small only.
- Use the same prompts for every model.
- Save the outputs before fine-tuning.
- Do not use the test split during baseline prompt creation.
