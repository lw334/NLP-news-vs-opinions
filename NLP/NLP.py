from flask import Flask, render_template, request
from emit_result import *
import settings

app = Flask(__name__)
settings.init()
prepare_clf()

@app.route("/", methods=['GET', 'POST'])
def render_result():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        text = request.form['sample']
        opinion = is_opinion(text)
        return render_template('index.html', text=text, opinion=opinion)

def is_opinion(text):
    print(settings.vectorizer)
    return predict_sample(text, settings.vectorizer,settings.clf)


if __name__ == "__main__":
    app.debug = True
    app.run()
