import os
import json
from pyltp import Segmentor, Postagger, Parser


def get_json_dict(src, src_fine, src_dep):
	ret_dict = {}
	for i in range(len(src)):
		tmp_fine = src_fine[i]
		tmp_dep = src_dep[i]
		ret_dict[i] = {}
		tmp1_dict = ret_dict[i]
		tmp1_dict["cont"] = src[i]
		tmp1_dict["word"] = {}
		tmp1_word = tmp1_dict["word"]
		for j in range(len(tmp_fine)):
			tmp1_word[j + 1] = {}
			tmp2_dict = tmp1_word[j + 1]
			tmp2_dict["cont"] = tmp_fine[j]
			tmp2_dict["parent"] = tmp_dep[j][0]
			tmp2_dict["relation"] = tmp_dep[j][1]
	return ret_dict


LTP_DATA_DIR = r'D:\ltp_data_v3.4.0'  # LTP模型目录路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径
segmentor = Segmentor(cws_model_path)  # 初始化实例
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径
postagger = Postagger(pos_model_path)
psr_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
parser = Parser(psr_model_path)
with open(r'Text.txt', 'r', encoding='utf-8') as fin:
	src = fin.read().split('\n')
words_list = []
arcs_list = []
for item in src:
	words = segmentor.segment(item)  # 分词
	word_list = list(words)
	words_list.append(word_list)
	postags = postagger.postag(words)  # 词性标注
	arcs = parser.parse(words, postags)  # 句法分析
	arcs_list.append(arcs)
print(words_list)
print(arcs_list)
json_dict = get_json_dict(src, words_list, arcs_list)
with open("ret2.json", 'w', encoding='utf-8') as fout:
	json.dump(json_dict, fout, ensure_ascii=False, indent=2)
segmentor.release()
postagger.release()
parser.release()
