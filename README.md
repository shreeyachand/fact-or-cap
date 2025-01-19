# Fake News Detection

## Shreeya Chand | sachand@usc.edu

#### two sentence summary

## Dataset
- [Fake News Detection Datasets (Kaggle)](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
- preprocessing steps and reasoning
- removing news dateline from real articles

## Model Development and Training
- model implementation choices, the training procedure, the conditions you settled on (e.g., hyperparameters), and discuss why these are a good set for your task
- lr and eps from docs
- scheduler w warmup
- sequence classification library

## Model Evaluation/Results
- accuracy & binary cross entropy loss (automatically determined by transformers library)

## Discussion
- How well does your dataset, model architecture, training procedures, and chosen metrics fit the task at hand? 
- Can your efforts be extended to wider implications, or contribute to social good? Are there any limitations in your methods that should be considered before doing so?
- If you were to continue this project, what would be your next steps?

- lowercase vs. uppercase
- other features key to detection
- include title?
- topic - world news etc
- max sequence length. solution: random subsequence? average predictions from multiple sequences?
- these fake news articles are pretty easy for humans to pick out - opinionated language etc. 


### Observations - Initial Training (2 epochs)
- All text is being classfied as 1 (real), with ~55-60% despite there being fewer real samples
- Also reflected by val accuracy: 48%
- Need to ensure randomized batches or at least similar class distribution in training/val

## Sources
- [RoBERTa - HuggingFace](https://huggingface.co/docs/transformers/model_doc/roberta)
- [Fine Tuning Roberta for Sentiment Analysis](https://colab.research.google.com/github/DhavalTaunk08/NLP_scripts/blob/master/sentiment_analysis_using_roberta.ipynb)
- [L4/L5 Solutions](https://colab.research.google.com/drive/1NrVDXktmixZuHIILBUujVp9nCwwGsPu8)