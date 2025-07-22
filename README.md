
# MisstepMath Dataset Generation

## Overview:
MisstepMath is a semi-synthetic dataset designed to train and evaluate AI models that simulate diverse student mistakes and teacher responses across various grade levels and math topics. This repository provides code to generate the dataset using structured curriculum prompts and large language models (LLMs), specifically GPT-4o.

## Folder Structure:
```bash
MisstepMath/
├── README.md
├── LICENSE
├── data/
│   ├── seed_data.csv
├── prompts/
│   ├── generation_prompts.md
├── evaluation/
│   ├── model_evaluation/
│   │   ├── results/
│   │   │   ├──
│   │   ├── TTR_SemanticSimilarity.py
│   ├── evaluate_models.py
│   ├── evaluation_metrics.md
│   └── evaluation_results.csv
├── notebooks/
│   ├── 1_dataset_generation.ipynb
│   ├── 2_fine_tuning_setup.ipynb
│   └── 3_RAG_baseline_setup.ipynb
├── scripts/
│   ├── rag_retriever.py
│   └── utils.py
```

## Usage Instructions:

1. Upload `seed_dataset.csv` (the seed data) into your working directory.
2. Open `1_dataset_generation.ipynb` in Google Colab or any Python environment.
3. Modify `k_class`, `k_topic`, and `challenge_types` to your desired setup.
4. Run the notebook to generate new student-teacher examples based on the curriculum and challenge type.
5. Output is saved as `generated_misstepmath_data.jsonl`.

## How It Works:
- For a given grade-topic-subtopic-challengeType, the script fetches related seed data.
- Prompts are designed to generate more diverse examples building on the existing ones.
- GPT-4o is used with system constraints and JSON schema.
- The output includes student mistake types, example problems, and detailed teacher responses.

## Reproducibility:
All generation prompts and logic are shared in the notebook. You can recreate the dataset or extend it by modifying the input configurations.

## License:
University of Virginia

## Contact:
https://github.com/LLMEducation/MisstepMath
