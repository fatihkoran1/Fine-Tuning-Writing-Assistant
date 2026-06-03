# Guarani Wikipedia Loading Test

I tested the `wikimedia/wikipedia` dataset with the subset `20231101.gn`.

The dataset loaded successfully in Python using the Hugging Face `datasets` library.

## Observed dataset structure

The dataset contains the following useful columns:

- `id`
- `url`
- `title`
- `text`

The Hugging Face dataset page showed approximately 5.52k rows for the `20231101.gn` subset.

## Sample inspection

I inspected several sample rows. The dataset contains real Guarani Wikipedia article text, for example articles such as:

- Ayvu rapyta
- Avañe'ẽ
- Ñorairõ Guasu Mokõiha
- Don Quijote de la Mancha
- Peteĩ

## Current conclusion

This dataset is technically usable as a training dataset candidate for the project.

It is not Guarani Mbya. It is Paraguayan Guarani. Because of that, I will not claim that the model represents Guarani Mbya directly.

Instead, I will use Guarani Mbya as the real-world inspiration and use Paraguayan Guarani Wikipedia to test the technical fine-tuning workflow with limited data.