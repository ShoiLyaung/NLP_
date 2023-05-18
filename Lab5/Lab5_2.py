import os
import re
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller


class LtpParser:
	def __init__(self):
		LTP_DIR = r"D:\Documents\UniversityFiles\Second\Spring\NLP_\Lab5\ltp_data_v3.4.0"
		self.segmentor = Segmentor(os.path.join(LTP_DIR, "cws.model"))
		self.postagger = Postagger(os.path.join(LTP_DIR, "pos.model"))
		self.parser = Parser(os.path.join(LTP_DIR, "parser.model"))
		self.recognizer = NamedEntityRecognizer(os.path.join(LTP_DIR, "ner.model"))
		self.labeller = SementicRoleLabeller(os.path.join(LTP_DIR, 'pisrl_win.model'))

	def build_parse_child_dict(self, words, postags):
		child_dict_list = []
		format_parse_list = []
		arcs = self.parser.parse(words, postags)  # 建立依存句法分析树
		print("分词列表：words = {}".format(words))
		rely_ids = [arc[0] - 1 for arc in
					arcs]  # 提取该句话的每一个词的依存父节点id
		heads = ['Root' if rely_id == -1 else words[rely_id] for rely_id in rely_ids]  # 匹配依存父节点词语
		relations = [arc[1] for arc in arcs]  # 提取依存关系

		for word_index in range(len(words)):
			child_dict = dict()  # 每个词语与所有其他词语的关系字典
			for arc_index in range(len(arcs)):
				# 当“依存句法分析树”遍历，遇到当前词语时，说明当前词语在依存句法分析树中与其他词语有依存关系
				if word_index == rely_ids[arc_index]:
					if relations[
						arc_index] in child_dict:
						child_dict[relations[arc_index]].append(arc_index)
					else:
						child_dict[relations[arc_index]] = []
						child_dict[relations[arc_index]].append(
							arc_index)
			child_dict_list.append(child_dict)  # 每个词对应的依存关系父节点和其关系
		# 整合每个词语的句法依存关系
		for i in range(len(words)):
			a = [relations[i], words[i], i, postags[i], heads[i], rely_ids[i] - 1, postags[rely_ids[i] - 1]]
			format_parse_list.append(a)
		return child_dict_list, format_parse_list

	# 语义角色标注
	def format_labelrole(self, words, postags):
		arcs = self.parser.parse(words, postags)  # 建立依存句法分析树
		roles = self.labeller.label(words, postags, arcs)
		roles_dict = {}
		for role in roles:
			# print(role)
			roles_dict[role[0]] = {arg[0]: [arg[0], arg[1][0], arg[1][1]] for arg in role[1]}
		return roles_dict

	def parser_main(self, sentence):
		words = list(self.segmentor.segment(sentence))
		postags = list(self.postagger.postag(words))
		child_dict_list, format_parse_list = self.build_parse_child_dict(words, postags)
		roles_dict = self.format_labelrole(words, postags)
		return words, postags, child_dict_list, format_parse_list, roles_dict


# 关系抽取类
class TripleExtractor:
	def __init__(self):
		self.parser = LtpParser()

	def split_sents(self, content):
		return [sentence for sentence in re.split(r'[？?！!。；;：:\n\r]', content) if sentence]

	def ruler1(self, words, postags, roles_dict, role_index):
		v = words[role_index]
		role_info = roles_dict[role_index]
		if 'A0' in role_info.keys() and 'A1' in role_info.keys():
			s = ''.join([words[word_index] for word_index in range(role_info['A0'][1], role_info['A0'][2] + 1) if
						 postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
			o = ''.join([words[word_index] for word_index in range(role_info['A1'][1], role_info['A1'][2] + 1) if
						 postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
			if s and o:
				return '1', [s, v, o]
		return '4', []

	def ruler2(self, words, postags, child_dict_list, format_parse_list, roles_dict):
		svos = []
		for index in range(len(postags)):
			tmp = 1
			# 借助语义角色标注的结果，进行三元组抽取
			if index in roles_dict:
				flag, triple = self.ruler1(words, postags, roles_dict, index)
				if flag == '1':
					svos.append(triple)
					tmp = 0
			if tmp == 1:
				if postags[index]:
					# 抽取以谓词为中心的事实三元组
					child_dict = child_dict_list[index]
					if 'SBV' in child_dict and 'VOB' in child_dict:
						r = words[index]
						e1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
						e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
						svos.append([e1, r, e2])

					relation = format_parse_list[index][0]
					head = format_parse_list[index][2]
					if relation == 'ATT':
						if 'VOB' in child_dict:
							e1 = self.complete_e(words, postags, child_dict_list, head - 1)
							r = words[index]
							e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
							temp_string = r + e2
							if temp_string == e1[:len(temp_string)]:
								e1 = e1[len(temp_string):]
							if temp_string not in e1:
								svos.append([e1, r, e2])
					# 含有介宾关系的主谓动补关系
					if 'SBV' in child_dict and 'CMP' in child_dict:
						e1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
						cmp_index = child_dict['CMP'][0]
						r = words[index] + words[cmp_index]
						if 'POB' in child_dict_list[cmp_index]:
							e2 = self.complete_e(words, postags, child_dict_list, child_dict_list[cmp_index]['POB'][0])
							svos.append([e1, r, e2])
		return svos

	def complete_e(self, words, postags, child_dict_list, word_index):
		child_dict = child_dict_list[word_index]
		prefix = ''
		if 'ATT' in child_dict:
			for i in range(len(child_dict['ATT'])):
				prefix += self.complete_e(words, postags, child_dict_list, child_dict['ATT'][i])
		postfix = ''
		if postags[word_index] == 'v':
			if 'VOB' in child_dict:
				postfix += self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
			if 'SBV' in child_dict:
				prefix = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix

		return prefix + words[word_index] + postfix

	def triples_main(self, text):
		sentences = self.split_sents(text)
		svos = []
		for index, sentence in enumerate(sentences):
			words, postags, child_dict_list, format_parse_list, roles_dict = self.parser.parser_main(sentence)
			svo = self.ruler2(words, postags, child_dict_list, format_parse_list, roles_dict)
			svos += svo
		return svos


def run_extractor(text):
	extractor = TripleExtractor()
	svos = extractor.triples_main(text)
	return svos


if __name__ == '__main__':
	for ran in range(5):
		src = "input" + str(ran+1) + ".txt"
		with open(src, 'r', encoding='utf-8') as f:
			text = f.read()
		print(text)
		ret = run_extractor(text)
		print("关系抽取结果：ret = {0}\n===============================\n".format(ret))
