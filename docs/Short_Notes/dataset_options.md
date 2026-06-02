# Dataset Options

## Option 1: FLORES+
Link: https://huggingface.co/datasets/openlanguagedata/flores_plus
Type: Parallel multilingual / low-resource translation dataset
Possible use: Select one low-resource language and limit the data to 1,000–3,000 examples.
Pros: Clean, documented, safer than sensitive Indigenous data, easy to explain.
Risks: Translation sentences may not fully represent natural writing assistant use.

## Option 2: AfroLingu-MT
Link: https://huggingface.co/datasets/UBC-NLP/AfroLingu-MT
Type: African low-resource translation dataset
Possible use: Select one African low-resource language pair and fine-tune on limited examples.
Pros: Strong low-resource fit.
Risks: More complex dataset structure and language choice.

## Option 3: Common Corpus
Link: https://huggingface.co/datasets/PleIAs/common_corpus
Type: Open/public multilingual text corpus
Possible use: Select safe licensed text and artificially limit it to simulate low-resource conditions.
Pros: Strong license/ethics argument.
Risks: Less directly connected to low-resource language preservation.

# Updated Initial Decision

My initial decision was to investigate FLORES+ because it is structured, multilingual, documented, and realistic for a 19-day student project.

However, after checking it more carefully, I found that FLORES+ is better suited for evaluation than training. Because of that, I will not use FLORES+ as my training dataset.

My updated dataset strategy is:

- Use Guarani Wikipedia (`wikimedia/wikipedia`, subset `20231101.gn`) as the training dataset candidate.
- Use FLORES+ only as a possible evaluation reference, if it fits the final setup.

I chose Guarani Wikipedia as the training candidate because it contains real Guarani text, has around 5.52k rows, and is more suitable for language-model fine-tuning than a translation benchmark.

I will not claim that this represents Guarani Mbya directly. Instead, I use Guarani Mbya as the real-world inspiration and use Paraguayan Guarani Wikipedia to test the technical question: can a small fine-tuned model adapt to limited examples, or does it mostly memorize?

## Training Dataset Candidate: Guarani Wikipedia

Dataset: wikimedia/wikipedia  
Subset: 20231101.gn  
Rows: approximately 5.52k  
Columns: id, url, title, text  
Language: Paraguayan Guarani  
Purpose: Training dataset candidate  

Why it fits:
- It contains real Guarani text.
- It has enough rows for a small low-resource fine-tuning experiment.
- It is more suitable for training than FLORES+, because FLORES+ is intended for evaluation.
- I can limit it to 1,000–3,000 examples to simulate a low-resource setting.

Limitations:
- It is Paraguayan Guarani, not Guarani Mbya.
- Wikipedia text is encyclopedic, not natural writing assistant text.
- I cannot fully judge language quality because I do not speak the language.
- Therefore, my evaluation will focus on adaptation, memorization, and generalization, not perfect linguistic correctness.

Current conclusion:
Guarani Wikipedia is a strong training dataset candidate. FLORES+ can still be considered for evaluation, but not for training.