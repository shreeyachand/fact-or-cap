from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

def process_text(text):
    model = AutoModelForSequenceClassification.from_pretrained('1-cased.pt')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=False)
    inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
    ids = torch.tensor([inputs['input_ids']])
    mask = torch.tensor([inputs['attention_mask']])
    out = model(input_ids=ids, attention_mask=mask)
    index = int(torch.argmax(out.logits, dim=1))
    conf = round(float(torch.softmax(out.logits, dim=1)[0][index]) * 100, 2)
    return index, conf


@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "POST" and request.form.get('text_box') is not '':
        text = request.form.get('text_box')
        real, con = process_text(text)
    else:
        real = ''
        con = ''
        text = ''
    return render_template("base.html", real=real, con=con, article = text)

if __name__ == '__main__':
    app.run(debug=True)