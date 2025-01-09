from flask import Flask, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

def process_text():
    model = AutoModelForSequenceClassification.from_pretrained('2-epochs.pt')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
    test_text = "Featured image"
    inputs = tokenizer.encode_plus(
            test_text,
            None,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
    ids = torch.tensor([inputs['input_ids']])
    mask = torch.tensor([inputs['attention_mask']])
    out = model(input_ids=ids, attention_mask=mask)
    print(torch.softmax(out.logits, dim=1))
    print(torch.argmax(out.logits, dim=1))


@app.route('/')
def home():
    process_text()
    return render_template("base.html")

if __name__ == '__main__':
    app.run(debug=True)