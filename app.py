import argparse
import datetime
import logging as rel_log
import os
import shutil
from datetime import timedelta
from flask import *

from infer.inference import predict
from init.GSRTR import GSRTRansfomer

UPLOAD_FOLDER = r'./static/input'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpge'])
app = Flask(__name__)

app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

werkzeug_logger = rel_log.getLogger('werkzeug')
werkzeug_logger.setLevel(rel_log.ERROR)

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


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
    verb = ''
    res_info = []
    my_file = request.files['file']
    print(datetime.datetime.now(), my_file.filename)

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
