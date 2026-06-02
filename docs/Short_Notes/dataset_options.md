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

# Initial Decision

My initial decision is to use FLORES+.

I chose FLORES+ because it is structured, multilingual, documented, and realistic for a 19-day student project. It allows me to select one language and limit the examples to simulate a low-resource setting.

I will not claim that this fully represents natural Guarani Mbya writing. Instead, I use Guarani Mbya as the real-world inspiration and use FLORES+ to test the technical question: can a small fine-tuned model adapt to limited examples, or does it mostly memorize?