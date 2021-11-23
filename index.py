# coding=utf-8
# 兼容python3
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os, zipfile
import globals as _g
import inputs
import sys
sys.path.insert(1, './src')
import model
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = './files'
MODEL_FOLDER = './saved_model'
ALLOWED_EXTENSIONS = set(['zip'])

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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

def check_execution_list(uploaded_dir_list):
    path = ''
    for execution_path in uploaded_dir_list:
        if 'Execution_lists.txt' in execution_path:
            path = execution_path

    if path == '':
        print("The 'Execution_lists.txt' does not exist.")
        raise Execution_error

    return path


def second_largest_number(arr):
    unique_nums = set(arr)
    sorted_nums = sorted(unique_nums, reverse=True)
    index = np.where(arr == sorted_nums[1])

    return sorted_nums[1], index[0][0]


class Execution_error(Exception):
    pass


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['error'] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


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
            uploaded_dir_list = unzip_file(filename)
            execution_list_path = ''
            try:
                execution_list_path = check_execution_list(uploaded_dir_list)
                print("execution_list_path", execution_list_path)
            except Exception as e:
                raise InvalidUsage("The 'Execution_lists.txt' does not exist.", status_code=400)

            #print("filename:", uploaded_dir_list)
            # view_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename.rstrip('.zip'), "Execution_lists.txt")
            view_filename = os.path.join(app.config['UPLOAD_FOLDER'], execution_list_path)
            print("view_filename", view_filename)

            dataset, steps, data_filename_list = 0, 0, 0
            try:
                dataset, steps, data_filename_list = inputs.prepare_dataset(view_filename)
            except Exception as e:
                raise InvalidUsage("The contents of the 'Execution_lists.txt' are incorrect.", status_code=400)

            crf_model = model.inference_multi_view_with_crf2()
            crf_model.load_weights(os.path.join(app.config['MODEL_FOLDER'], 'crf_95.h5'))
            crf_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=1e-5),
                              loss=tf.keras.losses.categorical_crossentropy,
                              metrics=[tf.keras.metrics.categorical_accuracy])
            try:

                predicted_results = crf_model.predict(dataset, steps=steps)

                element_id = []
                class_id_list = []
                class_name_list = []
                inference_class_id_list_2nd = []
                inference_class_name_list_2nd = []
                highest_softmax_list = []
                second_highest_softmax_list = []
                inference_class_name_list = []
                results = {}

                for filename in data_filename_list:
                    element_id.append(filename.split('/')[3])
                    class_name_list.append(filename.split('/')[2])

                for i in predicted_results:
                    class_id = np.argmax(i)
                    class_id_list.append(class_id)
                    highest_softmax_list.append(i.max())
                    second_highest_softmax, index = second_largest_number(i)
                    inference_class_id_list_2nd.append(index)
                    second_highest_softmax_list.append(second_highest_softmax)
                class_name = ['Beam', 'Column', 'Slab', 'Wall', 'Window', 'Stairflight', 'Member', 'Railing',
                              'Curtainwall', 'Covering', 'Singledoor', 'Doubledoor', 'Revolvingdoor']
                #
                for i in class_id_list:
                    inference_class_name_list.append(class_name[i])
                for i in inference_class_id_list_2nd:
                    inference_class_name_list_2nd.append(class_name[i])

                for i in range(len(element_id)):
                    tmp = []
                    tmp.append({'inference_class_id_1st': str(class_id_list[i]),
                                'inference_class_name_1st': inference_class_name_list[i],
                                'true_class': class_name_list[i], 'softmax_1st': str(highest_softmax_list[i]),
                                'softmax_2nd': str(second_highest_softmax_list[i]),
                                'inference_class_name_2nd': inference_class_name_list_2nd[i],
                                'inference_class_id_2nd': str(inference_class_id_list_2nd[i]),
                                'file_path': data_filename_list[i].rstrip('/ImageList.txt')})
                    results[element_id[i]] = tmp

                return json.dumps(results, indent=4, ensure_ascii=False)
            except Exception as e:

                raise InvalidUsage("It cannot be inferred. Please check the file again.", status_code=400)

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
    app.config['MODEL_FOLDER'] = MODEL_FOLDER
    app.run(host='0.0.0.0', port=3000)