from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
import torch
import os
from werkzeug.utils import secure_filename
import pandas as pd
import sys
sys.path.append('../')
import NeuralNetDefinitions as nets


# declare constants
HOST = '0.0.0.0'
PORT = 8081

# initialize flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = '12345'
app.config['UPLOAD_FOLDER'] = "images"

#loading neural nets

hazelarea = nets.HazelNet().to(nets.device)
hazelarea.load_state_dict(torch.load(f"../bestmodel_area"))

@app.route('/', methods=['GET', 'POST'])
def root():

    df = pd.read_csv("logging") if (os.path.isfile("logging")) else pd.DataFrame()

    if request.method == 'POST':
        image = secure_filename(request.files["image"].filename)
        template = secure_filename(request.files["template"].filename)
        if image == '' or template == '':
            flash("There is data missing!", "error")
            return render_template("index.html")

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], image)
        template_path = os.path.join(app.config['UPLOAD_FOLDER'], template)
        label = request.form["text"]
        pred  = 0

        request.files["image"].save(img_path)
        request.files["template"].save(template_path)

        pred = evalimgs(template_path, img_path)

        results = {"label" : label, "HazelArea" : pred}

        messages = {"success_text" : "File Sucessfully Send", 
                    "results" :  results
                    }

        results = {"img" : img_path, "template" : template_path} | results
        
        df = df.append(results, ignore_index=True)
        df.to_csv("logging", index=False)


        flash(messages, "success")
    return render_template("index.html")

@app.route('/results', methods=['GET'])
def results():
    df = pd.read_csv("logging").sort_index(ascending=False)
    return render_template("results.html", tables=[df.to_html(classes='data', header='true')])

if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    # run web server
    app.run(host=HOST,
            debug=True,
            port=PORT)

app


def evalimgs(template, img):
    template, img = nets.readImg_url(template, img)
    pred = nets.predict(hazelarea, template, img)
    return pred