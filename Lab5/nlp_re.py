import opennre

model = opennre.get_model('wiki80_bert_softmax')

input_test = {
    'text':'Blowing in the wind is a song composed by Bob Dylan who is a songwriter.',
    'h':{
        'pos':(42,51)
    },
    't':{
        'pos':(61,71)
    }
    }

with open('example.txt', 'r') as file_in, open('out/re.out', 'w') as file_out:
    inputs = eval(file_in.read())
    for input in inputs:
        print(input["text"])
        print('(%s, %s)' % (input["text"][input['h']['pos'][0]:input['h']['pos'][1]],input["text"][input['t']['pos'][0]:input['t']['pos'][1]]))
        output = model.infer(input)
        print(output)
        file_out.write('(%s, %s)' % (input["text"][input['h']['pos'][0]:input['h']['pos'][1]],input["text"][input['t']['pos'][0]:input['t']['pos'][1]]))
        file_out.write(str(output)+'\n')

# print(model.infer(input_test))

