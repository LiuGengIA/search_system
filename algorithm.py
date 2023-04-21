import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

import copy
import numpy as np

from search_system.clip import clip
# 运行algorithm不知道search_system，运行app就知道，神奇
# from clip import clip

# from models.blip_retrieval import blip_retrieval
import torch.nn.init as init
import os
import pickle
from PIL import Image

import json
import xlrd
# import ruamel_yaml as yaml

def standard_json(json_sict): # 对json文件进行规范，key按顺序排列，字符串统一小写
    index = 0
    standard_json_dict = {}
    for id in json_sict:
        standard_json_dict[str(index)] = {}
        standard_json_dict[str(index)]["id"] = str(index)
        standard_json_dict[str(index)]["name"] = json_sict[str(index)]["name"].lower()
        standard_json_dict[str(index)]["description"] = json_sict[str(index)]["description"].lower()
        standard_json_dict[str(index)]["path"] = json_sict[str(index)]["path"].lower()
        standard_json_dict[str(index)]["local_path"] = json_sict[str(index)]["local_path"]
        standard_json_dict[str(index)]["attribute_tag"] = json_sict[str(index)]["attribute_tag"]
        standard_json_dict[str(index)]["style_tag"] = json_sict[str(index)]["style_tag"]
        standard_json_dict[str(index)]["category_tag"] = json_sict[str(index)]["category_tag"]
        standard_json_dict[str(index)]["color_tag"] = json_sict[str(index)]["color_tag"]
        standard_json_dict[str(index)]["shape_tag"] = json_sict[str(index)]["shape_tag"]
        standard_json_dict[str(index)]["texture_tag"] = json_sict[str(index)]["texture_tag"]
        index += 1
        
    return standard_json_dict
    

def convert_xlsx_to_json(xlsx_path, current_json_path=None):
    data = xlrd.open_workbook(xlsx_path)
    if current_json_path:
        with open(current_json_path, "r+") as f:
            json_dict = json.load(f)
    else:
        json_dict = {}

    for index in range(7):
        table = data.sheet_by_index(index)
        name = table.name.lower()
        nrows = table.nrows
        ncols = table.ncols

        current_id = len(json_dict.keys())
        for i in range(nrows-1):
            json_dict[str(current_id+i)] = {}
            json_dict[str(current_id+i)]["id"] = str(current_id+i)
            json_dict[str(current_id+i)]["name"] = table.cell_value(i+1, 0).lower()
            json_dict[str(current_id+i)]["description"] = table.cell_value(i+1, 1).lower()
            json_dict[str(current_id+i)]["path"] = table.cell_value(i+1, 2)
            json_dict[str(current_id+i)]["local_path"] = "./static/sample_images/{0}/{1}{2}.png".format(name, name, i+1)
            json_dict[str(current_id+i)]["attribute_tag"] = {}
            json_dict[str(current_id+i)]["style_tag"] = {}
            json_dict[str(current_id+i)]["category_tag"] = {}
            json_dict[str(current_id+i)]["color_tag"] = {}
            json_dict[str(current_id+i)]["shape_tag"] = {}
            json_dict[str(current_id+i)]["texture_tag"] = {}


    if not current_json_path:
        current_json_path = "samples_all.json"
    with open(current_json_path[:-5]+"_standard"+current_json_path[-5:], "w+") as f:
        json.dump(standard_json(json_dict), f, indent=2)
    return json_dict


class retrieval_demo:
    def __init__(self):
        self.CLIP = self.load_clip_to_cpu("ViT-B/16").cuda()
        self.dtype = self.CLIP.dtype
        self.image_encoder = self.CLIP.visual  # image.type(self.dtype)
    
    def load_clip_to_cpu(self, name):
        backbone_name = name
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
        # model_path = "/data1/geng_liu/retrieval/model_base.pth"
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cuda").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
            
        model = clip.build_model(state_dict or model.state_dict())
        # model = clip.build_model(state_dict["model"])

        return model.float()
    
    def load_json(self, path):
        with open(path, "r") as f:
            self.asset_dict = json.load(f)
        return self.asset_dict
    
    def load_features(self, extract_path):
        with open(os.path.join(extract_path, "image_features.pkl"), "rb") as f:
            self.image_features = pickle.load(f)
            self.n_image_features = self.image_features.shape[0]
    
    def extract_features(self, json_path, extract_path): 
        # 按照顺序load json文件中的图像
        # 完了可能需要设置batch size
        image_list = []
        with open(json_path, "r") as f:
            json_dict = json.load(f)
            for id in json_dict:
                # image_path = json_dict[id]["local_path"]
                image_list.append(self.image_preprocess(json_dict[id]["local_path"]))
                
        images = torch.cat(image_list, 0).cuda()
        image_features = self.image_encoder(images)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        
        with open(os.path.join(extract_path, "image_features.pkl"), "wb") as f:
            pickle.dump(image_features, f)
        
        return image_features
                
    
    def retrieval_str(self, indicator, input_str): 
        # indicator指出要检索什么类型，包括name，description，tag
        # str是要检索的字符串
        # 暂时先输出一些id吧
        result = []
        if "tag" in indicator:
            for id in self.asset_dict:
                for key in self.asset_dict[id]:
                    if indicator in key:
                        for tag_id in self.asset_dict[id][key]:
                            if input_str in self.asset_dict[id][key][tag_id]:
                                result.append(self.asset_dict[str(id)])
                                break
        else:
            for id in self.asset_dict:
                if input_str in self.asset_dict[id][indicator]:
                    result.append(self.asset_dict[str(id)])
        return result
        
    def text_preprocess(self, text): # 把文本变成可输入模型的
        return clip.tokenize(text).cuda()
    
    def image_preprocess(self, image_path): # 把图像变成tensor
        preprocess = clip._transform(self.CLIP.visual.input_resolution)
        return preprocess(Image.open(image_path)).unsqueeze(0).cuda()
        
    def retrieval_semantic(self, indicator, input, num=10): # input是str的list，num是给出结果的数量
        # 可能需要两个indicator 分别是输入和输出的格式
        if indicator == "text":
            text = self.text_preprocess(input)
            retrieval_feature = self.CLIP.encode_text(text)
        elif indicator == "image":
            image = self.image_preprocess(input)
            retrieval_feature = self.image_encoder(image)
        else:
            assert 0, "unkonwn indicator!"
        
        # print(retrieval_feature.shape)
        retrieval_feature = retrieval_feature/retrieval_feature.norm(dim=-1, keepdim=True)
        image_similarities = retrieval_feature@self.image_features.t() # 1*N
        
        
        similarity_list = list(image_similarities.cpu().detach().view(-1).numpy())
        index_list = range(self.n_image_features)
        index_similarity_list = sorted(zip(index_list, similarity_list), key=lambda x:x[1], reverse=True)
        
        # print(index_similarity_list[:num])
        if len(index_similarity_list) >= num:
            return self.convert_id_to_dict(index_similarity_list[:num])
        else:
            return self.convert_id_to_dict(index_similarity_list)
        
    def convert_id_to_dict(self, ids):
        dict_list = []
        for id,similarity in ids:
            dict_list.append(self.asset_dict[str(id)])
            
        return dict_list
        
        
        # 根据相似度排序
            
        
            
        
        
        
def main():
    image_path = "/data1/geng_liu/retrieval/test_images/test_dog.png" # 示例图像的路径
    text = ["bnana"]
    demo = retrieval_demo()
    #demo.load_json()
    # demo.extract_features("/data1/geng_liu/search_system/json_files/samples_all_standard.json", \
    #     "/data1/geng_liu/search_system/extracted_features")
    demo.load_features("/data1/geng_liu/search_system/extracted_features")
    
    # print(demo.retrieval_str("name", text))
    # print(demo.retrieval_semantic("text", text))
    # print(demo.retrieval_semantic("image", image_path))

if __name__ == "__main__":
    # convert_xlsx_to_json("/data1/geng_liu/search_system/json_files/class2.xls", \
    #     "/data1/geng_liu/search_system/json_files/samples_all.json")
    main()