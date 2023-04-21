from search_system.algorithm import *
from flask import Flask, request, render_template
import json

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'
demo = retrieval_demo()
demo.load_features("/data1/geng_liu/search_system/extracted_features")
demo.load_json("/data1/geng_liu/search_system/json_files/samples_all_standard.json")


# 读取数据文件
# with open('/data1/geng_liu/search_system/json_files/samples_all.json', 'r') as f:
#     data = json.load(f)


# # 定义检索算法
# def search_data(search_type, search_content, search_image=None):
#     print(search_type)
#     print(search_image)
#     result = []
#     for item in data:
#         if search_content in item.get(search_type, ''):
#             result.append(item)
#     return result # json dict的列表


# 定义API接口
@app.route('/search', methods=['GET', 'POST'])
def search():
    search_type = ''
    search_content = ''
    if request.method == 'POST':
        search_type = request.form.get('search-type').lower()
        search_content = request.form.get('search-content').lower()
        if request.form.get('num'):
            num = request.form.get('num')
        else: num = 10
        if search_type == "image":
            search_image = request.files.get('search-image')
            save_path = "search.png"
            search_image.save(save_path)
            results = demo.retrieval_semantic("image", save_path, int(num))
        elif search_type == "text":
            results = demo.retrieval_semantic("text", search_content, int(num))
        else:
            results = demo.retrieval_str(search_type, search_content)
    else:
        results = []
    # print(results)
    # results[0]["path"] = "/data1/geng_liu/retrieval/test_images/test_dog.png"
    return render_template('index.html', results=results, search_type=search_type, search_content=search_content)


if __name__ == '__main__':
    app.run(debug=True)
