from gensim.models import Word2Vec
import hanlp
import numpy as np


def w2v(cont, mod):
	ret = np.zeros(100)
	for i in cont:
		try:
			ret += np.array(mod.wv[i])
		except:
			pass
	return ret / len(cont)


HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
model = Word2Vec.load('word2vec.model')
sentence1 = '这个苹果真好吃，多吃苹果对身体好'
sentence2 = '一天一苹果，医生远离我'
sentence3 = '这台苹果手机的性能挺不错的'
sentence4 = '俄罗斯反垄断监管机构开始对苹果公司展开调查'
sentences_apple = [sentence1, sentence2, sentence3, sentence4]
infer1 = '美国一家高科技公司，经典产品有iPhone'
infer2 = '水果的一种，营养价值很高'
infer_apple = [infer1, infer2]
des1 = HanLP(infer1).to_dict()['tok/fine']
des2 = HanLP(infer2).to_dict()['tok/fine']
print("\"苹果\"-释义：\n1.", infer1, "\n2.", infer2)
for sentence in sentences_apple:
	sen1 = HanLP(sentence).to_dict()['tok/fine']
	content = w2v(sen1, model)
	d1 = w2v(des1, model)
	d2 = w2v(des2, model)
	if np.dot(d1, content) / (np.linalg.norm(d1) * (np.linalg.norm(content))) >= np.dot(d2, content) / (
			np.linalg.norm(d2) * (np.linalg.norm(content))):
		print(sentence, " --- 释义1：", infer1)
	else:
		print(sentence, " --- 释义2：", infer2)
print("---------------------------------------------------------")
sentence1 = '会计们正在为公司算账'
sentence2 = '收银正在算账核对金额中'
sentence3 = '恼羞成怒，威胁要和他算账。'
sentence4 = '我非找他算账不可，怨恨在胸中滋生着，气恨难忍了'
infer1 = '计算经济财务的账目'
infer2 = '吃亏失败后找人争执'
sentences_sz = [sentence1, sentence2, sentence3, sentence4]
infer_sz = [infer1, infer2]
print("\"算账\"-释义：\n1.", infer1, "\n2.", infer2)
for sentence in sentences_sz:
	sen1 = HanLP(sentence).to_dict()['tok/fine']
	content = w2v(sen1, model)
	d1 = w2v(des1, model)
	d2 = w2v(des2, model)
	if np.dot(d1, content) / (np.linalg.norm(d1) * (np.linalg.norm(content))) >= np.dot(d2, content) / (np.linalg.norm(d2) * (np.linalg.norm(content))):
		print(sentence, " --- 释义1：", infer1)
	else:
		print(sentence, " --- 释义2：", infer2)
