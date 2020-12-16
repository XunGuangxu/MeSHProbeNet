import random

output_file = 'train.tsv'

wf = open(output_file, 'w')
for i in range(1000):
    words = [random.randint(1, 1000) for j in range(random.randint(1, 100))]
    jrnl = random.randint(1, 1000)
    meshs = [random.randint(1, 1000) for j in range(random.randint(1, 20))]
    wf.writelines(' '.join([str(w) for w in words]) + '\t' +
                  str(jrnl) + '\t' +
                  ' '.join([str(m) for m in meshs]) + '\n')
wf.close()
print('done.')