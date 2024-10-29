
file = open('main_revised.tex','r').readlines()
i=0
while i < len(file):
    line = file[i].strip()
    if line and line[0] == '%':
        file.pop(i)
    else:
        i+=1
file = '\n'.join(file)


citations = set()

while (
        ((i:= file.find(r'\cite{')) > -1) |
        ((j:= file.find(r'\textcite{')) > -1)
    ):
    if j == -1 or (i!=-1 and i < j):
        idx = i + len(r'\cite{')
    else:
        idx = j + len(r'\textcite{')
    file = file[idx:]
    refs = file[:file.find('}')].split(',')
    refs = [ref.strip() for ref in refs]
    citations.update(set(refs))

citations = list(citations)
citations.sort()
open('main_revised_citations.txt','w').write('\n'.join(citations))

