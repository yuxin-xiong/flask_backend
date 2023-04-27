import argparse
import datetime
import logging as rel_log
import os
import shutil
from datetime import timedelta
from flask import *

from infer.inference import predict
from init.GSRTR import GSRTRansfomer
from flask_sqlalchemy import SQLAlchemy

UPLOAD_FOLDER = r'./static/input'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpge'])
app = Flask(__name__)

app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:04036837@localhost/gsrtr'
db = SQLAlchemy(app)

werkzeug_logger = rel_log.getLogger('werkzeug')
werkzeug_logger.setLevel(rel_log.ERROR)

class ImageFIle(db.Model):
    filename = db.Column(db.String(255), primary_key=True)
    verb = db.Column(db.String(255))

@app.route('/search',methods=['GET','POST'])
def search_imag():
    keyword = request.args.get('keyword')

    results = ImageFIle.query.filter(ImageFIle.verb.like(f'%{keyword}%')).all()


# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    # print(current_app.model.args)
    return redirect(url_for('static', filename='./index.html'))


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    res_info = []
    my_file = request.files['file']

    if my_file and allowed_file(my_file.filename):
        # 合并路径
        src_path = os.path.join(app.config['UPLOAD_FOLDER'], my_file.filename)
        print(src_path)  # ./image/input\basketball.jpg
        # 保存图片
        my_file.save(src_path)
        args = current_app.args
        print('-' * 20 + " ready predict " + '-' * 20)
        label_color = predict(args.model, device=args.device, image_path=src_path, inference=True,
                            idx_to_verb=args.idx_to_verb, idx_to_role=args.idx_to_role,
                            vidx_ridx=args.vidx_ridx, idx_to_class=args.idx_to_class,
                            output_dir=args.output_dir)

        # 信息提取
        res_path = 'static/output/' + "{}_result.txt".format(my_file.filename.split('.')[0])
        with open(res_path, "r") as file:
            content = file.read()
        lines = content.split("\n")
        # 当前行为动词
        verb = lines[0].split(":")[1].strip()

        for line in lines:
            if line.strip() == "":
                continue
            elif 'role:' in line and 'noun:' in line:
                role = line.split(',')[0].split(':')[1].strip()
                noun_t = line.split(',')[1].split(':')[1].strip()
                noun = noun_t.split('.')[0]
                color = label_color.get(noun, 'null')
                print(color)
                res_info.append({'role': role, 'noun': noun, 'color': color})
        print(res_info)

        return jsonify({'status': 1,
                        'image_url': 'http://127.0.0.1:5000/static/input/' + my_file.filename,
                        'draw_url': 'http://127.0.0.1:5000/static/output/' + "{}_result.jpg".format(
                            my_file.filename.split('.')[0]),
                        'verb': verb,
                        'res_info': res_info
                        })
    return jsonify({'status': 0})


if __name__ == '__main__':
    with app.app_context():
        current_app.args = GSRTRansfomer().args
    app.run(host='127.0.0.1', port=5000, debug=False)
