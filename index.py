from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os, zipfile
import globals as _g
import inputs
from src import model as m

app = Flask(__name__)
UPLOAD_FOLDER = './files'

ALLOWED_EXTENSIONS = set(['zip'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Existing the directory")

def unzip_file(filename):
    uploaded_zipfile = zipfile.ZipFile(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    uploaded_zipfile.extractall(os.path.join(app.config['UPLOAD_FOLDER'], filename.rstrip('.zip')))
    uploaded_zipfile.close()

@app.route('/upload')
def render_file():
    return render_template('upload.html')

@app.route('/fileUpload', methods=['POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            unzip_file(filename)

            return "done"

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.run(host='0.0.0.0', port=3000)
