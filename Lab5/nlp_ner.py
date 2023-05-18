import hanlp

hanlp.pretrained.mtl.ALL
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)

with open('example.txt', 'r',encoding='utf-8') as file_in, open('hanlp.json', 'w') as file_out:
    sents = file_in.read()
    sents = sents.split()
    # sent = '杭州高新技术产业开发区成果显著'
    HanLP(sents, tasks='ner').pretty_print()
    file_out.write(str(HanLP(sents, tasks='ner'))+'\n')