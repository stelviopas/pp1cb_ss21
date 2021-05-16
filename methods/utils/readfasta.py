
def readfasta(seq_file, labels_file):
    #seq_file = input('please enter the sequence file path')
    #labels_file = input('please enter the labals file path')
    seq_d = {}
    seq_number = 0
    label_number =0
    with open(seq_file, 'r',newline='') as reader:
        seq_num = ''
        for line in reader.readlines():
            line = line.rstrip()
            if line.startswith('>'):
                seq_num = line[1:]
                seq_number+=1
            else:
                seq_d[seq_num] = [line]
        #print(seqnum_d)

    with open(labels_file, 'r',newline='') as reader:
        label_num = ''
        for line in reader.readlines():
            line = line.rstrip()
            if line.startswith('>'):
                label_number+=1
                label_num = line[1:]
            else:
                if label_num in seq_d:
                    seq_d[label_num].append(line)
                else:
                    seq_d[label_num] = ['', line]
    print("Read Done With %s Reads and %s Labels" %(seq_number,label_number))
    return seq_d
if __name__ == "__main__":
    merged = readfasta("D:\\Study thingie\\FACH\\S4 WS2021\\PP\\uebung\\data\\disorder_seq.fasta","D:\\Study thingie\\FACH\\S4 WS2021\\PP\\uebung\\data\\disorder_labels.fasta")
