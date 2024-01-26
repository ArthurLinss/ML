from flask import Flask, request, render_template, jsonify
import spacy
from flask_cors import CORS


# checkout: python -m spacy download en_core_web_sm


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
nlp = spacy.load("en_core_web_sm")

@app.route('/ner')
def index():
    return render_template('ner.html', title="Named Entity Recognition")



@app.route('/process_text', methods=['POST'])
def process_text():
    if request.method == 'POST':
        input_text = request.form['input_text']
        style = request.form['style']  # "dep", "ent"
        doc = nlp(input_text)
        html = render_displacy(doc, style)
        return html

@app.route('/get_ner_labels', methods=['POST'])
def get_ner_labels():
    if request.method == "POST":
        labels= nlp.get_pipe('ner').labels
        labels_info = []
        for label in labels:
            labels_info.append("%s: %s" % (label,spacy.explain(label)))
        return jsonify(labels_info), 200


def render_displacy(doc, style):
    options = {"compact": True, "bg": "#09a3d5", "color": "white", "font": "Source Sans Pro"}

    if style == 'ent':
        return spacy.displacy.render(doc, style='ent', options=options)
    elif style == 'dep':
        return spacy.displacy.render(doc, style='dep', options=options)

@app.route('/dropdown_values')
def dropdown_values():
    display_styles = ['ent', 'dep']
    return jsonify({'dropdownValues': display_styles})

if __name__ == '__main__':
    app.run(debug=True)

    # in terminal: flask --app api.py --debug run -p 8000

