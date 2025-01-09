# Fake News Detection

## Sources
- [RoBERTa - HuggingFace](https://huggingface.co/docs/transformers/model_doc/roberta)
- [Fine Tuning Roberta for Sentiment Analysis](https://colab.research.google.com/github/DhavalTaunk08/NLP_scripts/blob/master/sentiment_analysis_using_roberta.ipynb)
- [L4/L5 Solutions](https://colab.research.google.com/drive/1NrVDXktmixZuHIILBUujVp9nCwwGsPu8)

### Observations - Initial Training (2 epochs)
- All text is being classfied as 1 (real), with ~55-60% despite there being fewer real samples
- Also reflected by val accuracy: 48%
- Need to ensure randomized batches or at least similar class distribution in training/val