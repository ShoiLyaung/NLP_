import hanlp

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
with open(r'哈工大语料.txt', 'r', encoding='utf-8') as fin:
	src = fin.read().split('\n')
hlp = HanLP(src).to_dict()
# print(hlp)
srl = hanlp.load('CPB3_SRL_ELECTRA_SMALL')
for i in hlp['tok/fine']:
	if i:
		print(srl(i))
