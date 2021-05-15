seq_file = input('please enter the sequence file path')
labels_file = input('please enter the labals file path')
seq_d = {}

with open(seq_file, 'r',newline='') as reader:
    seq_num = ''
    for line in reader.readlines():
        line = line.rstrip()
        if line.startswith('>'):
            seq_num = line[1:]
        else:
            seq_d[seq_num] = [line]
    #print(seqnum_d)

with open(labels_file, 'r',newline='') as reader:
    label_num = ''
    for line in reader.readlines():
        line = line.rstrip()
        if line.startswith('>'):
            label_num = line[1:]
        else:
            if label_num in seq_d:
                seq_d[label_num].append(line)
            else:
                seq_d[label_num] = ['', line]
#print(seq_d["16364"])