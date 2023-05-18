import hanlp
import json


def get_json_dict(src, src_dict):
	ret_dict = {}
	src_dep = src_dict["dep"]
	src_fine = src_dict["tok/fine"]
	for i in range(len(src_dep)):
		ret_dict[i] = {}
		ret_dict[i]["cont"] = src[i]
		ret_dict[i]["word"] = {}
		tmp1_word = ret_dict[i]["word"]
		for j in range(len(src_fine[i])):
			tmp1_word[j + 1] = {}
			tmp1_word[j + 1]["cont"] = src_fine[i][j]
			tmp1_word[j + 1]["parent"] = src_dep[i][j][0]
			tmp1_word[j + 1]["relation"] = src_dep[i][j][1]
	return ret_dict


HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
with open(r'Text.txt', 'r', encoding='utf-8') as fin:
	src = fin.read().split('\n')
print(src)
# print(HanLP(src))
# ret_dict = HanLP(src).to_dict()
# json_dict = get_json_dict(src, ret_dict)
# with open("ret1.json", 'w', encoding='utf-8') as fout:
# 	json.dump(json_dict, fout, ensure_ascii=False, indent=2)
