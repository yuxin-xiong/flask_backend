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
import pymysql
UPLOAD_FOLDER = r'./static/input'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpge'])
app = Flask(__name__)

app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:04036837@localhost:3306/gsrtr'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#
# db = SQLAlchemy(app)
# cursor = db.cursor()

werkzeug_logger = rel_log.getLogger('werkzeug')
werkzeug_logger.setLevel(rel_log.ERROR)

# class fileverb(db.Model):
#     filename = db.Column(db.String(255), primary_key=True)
#     verb = db.Column(db.String(255))

conn = pymysql.connect(host='localhost', user='root', password='04036837', port=3306,
                       db='gsrtr')
cur = conn.cursor()





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
def transform_search_result(data, roles):
    roles.insert(0,'file')
    result = []
    for item in data:
        item_dict = {}
        for i, role in enumerate(roles):
            if role == 'file':
                item_dict[role] = item[i]
            else:
                item_dict[role] = item[i].split('.')[0]
        result.append(item_dict)
    return result
def transform_result(data, roles):
    roles.insert(0,'file')
    roles.insert(1,'verb')
    result = []
    for item in data:
        item_dict = {}
        for i, role in enumerate(roles):
            if role == 'file':
                item_dict[role] = item[i]
            else:
                item_dict[role] = item[i].split('.')[0]
        item_dict['relevantNum'] = item[-1]
        result.append(item_dict)
    return result

def build_select(verb, roles, nouns):

    # 构建主SELECT语句的基本部分
    main_select_clause = "SELECT file, verb"
    for role in roles:
        main_select_clause += f", {role}"
    main_select_clause += ', match_count'
    # 构建子SELECT语句的计算部分（match_count）
    match_count_calculation = ""
    match_count_calculation+= f"(verb = '{verb}') + "
    for i, role in enumerate(roles):
        if nouns[i] == "'blank'":
            nouns[i] = "blank"
        match_count_calculation += f"({role} like '%{nouns[i]}%') + "
    match_count_calculation = match_count_calculation[:-3]  # 移除最后的 "+ "

    sub_select_clause = "(SELECT file, verb"
    for role in roles:
        sub_select_clause += f", {role}"
    sub_select_clause += ', '
    sub_select_clause += match_count_calculation
    sub_select_clause += ' AS match_count\n'
    sub_select_clause += f"FROM rolestable WHERE verb = '{verb}'"
    sub_select_clause += ') AS subquery'

    # 构建WHERE语句的条件部分
    where_conditions = f"WHERE verb = '{verb}' OR "
    for i, role in enumerate(roles):
        if i > 0:
            where_conditions += " OR "
        where_conditions += f"{role} like '%{nouns[i]}%'"



    # 构建完整的目标字符串
    target_string = f"{main_select_clause}\n"
    target_string += f"FROM {sub_select_clause}\n"
    target_string += f"{where_conditions}\n"
    target_string += "ORDER BY verb DESC, match_count DESC\n"
    target_string += "LIMIT 20;"
    return target_string

def recommend(verb,roles,nouns):
    query = build_select(verb,roles,nouns)
    print("===================select query================")
    print(query)
    db_res = ''
    try:
        cur.execute(query)
        db_res = cur.fetchall()
    except:
        conn.rollback()

    print("===================query res================")
    print('db_res:',db_res)

    file_items = [item[0] for item in db_res]
    res_dic= transform_result(db_res, roles)

    print('file_items:'+ str(file_items))
    print('res_dic:'+ str(res_dic))
    return file_items, res_dic

# 上传图片进行预测
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
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

        roles = []
        nouns = []
        # 识别结果信息
        res_info = []
        for line in lines:
            if line.strip() == "":
                continue
            elif 'role:' in line and 'noun:' in line:
                role = line.split(',')[0].split(':')[1].strip()
                roles.append(role)
                noun_t = line.split(',')[1].split(':')[1].strip()
                noun = noun_t.split('.')[0]
                nouns.append(noun)
                color = label_color.get(noun, 'null')
                res_info.append({'role': role, 'noun': noun, 'color': color})
        print("resInfo:",res_info)

        # 推荐相关图片信息
        recommend_file_list, recommend_file_info = recommend(verb,roles,nouns)
        roles.append('relevantNum')
        print("roles:",roles)
        return jsonify({'status': 1,
                        'image_url': 'http://127.0.0.1:5000/static/input/' + my_file.filename,
                        'draw_url': 'http://127.0.0.1:5000/static/output/' + "{}_result.jpg".format(my_file.filename.split('.')[0]),
                        'verb': verb,
                        'res_info': res_info,
                        'recommend_file_list':recommend_file_list,
                        'recommend_file_info':recommend_file_info,
                        'recommend_file_roles':roles})

    return jsonify({'status': 0})

# 关键字搜索
@app.route('/search',methods=['GET','POST'])
def search_imag():
    keyword = request.args.get('keyword')
    print(keyword)

    idx_to_verb = current_app.args.idx_to_verb
    vidx_ridx = current_app.args.vidx_ridx
    idx_to_role = current_app.args.idx_to_role

    verb_index = idx_to_verb.index(keyword)
    roles_index = vidx_ridx[verb_index]
    column_labels= []
    for i in range(len(roles_index)):
        column_labels.append(idx_to_role[roles_index[i]])

    print("roles_index: ", roles_index)
    print("column_labels: ", column_labels)

    select_caluse = 'SELECT file'
    for col in column_labels:
        select_caluse += f", {col}"
    print(select_caluse)
    query = select_caluse + f" FROM rolestable WHERE verb like '%{keyword}%' LIMIT 40"
    print("===================select query================")
    print(query)
    res = ''
    try:
        cur.execute(query)
        res = cur.fetchall()
    except:
        conn.rollback()
    print(res)

    file_items = [item[0] for item in res]
    res_dic=transform_search_result(res, column_labels)

    print('file_items:'+ str(file_items))
    print('res_dic:'+ str(res_dic))

    return jsonify({
        'search_file_list':file_items,
        'search_file_info':res_dic,
        'search_file_roles':column_labels
    })


if __name__ == '__main__':
    with app.app_context():
        current_app.args = GSRTRansfomer().args
    app.run(host='127.0.0.1', port=5000, debug=False)
