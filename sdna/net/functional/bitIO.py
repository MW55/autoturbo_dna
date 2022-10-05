import os
import numpy as np

from sdna.net.functional import utils


class BitEncodeStream(object):
    def __init__(self, inp_, outp_, blocksize, index_size):
        self.input = inp_
        self.output = outp_
        self.blocksize = blocksize
        self.index_size = index_size
        self.padding = self.get_padding()
        #self.blocks = (self.size*8)//self.blocksize
        self.index_counter = 0
        self.curr_block = np.zeros(self.blocksize, dtype=np.float32)
        #readbuffer = np.array()

    def get_padding(self):
        size = os.path.getsize(self.input) * 8
        payload_size = (self.blocksize - self.index_size)
        rem = size % payload_size
        if rem:
            return payload_size - rem
        else:
            return 0

    def read(self, args, net):
        with open(self.input, "rb") as inp, open(self.output, "w") as out:
            self.add_index()
            c_byte = inp.read(1)
            start = self.index_size + self.padding
            end = start + 8
            while c_byte != b'':
                self.insert_data(start, end, c_byte)
                start += 8
                end += 8
                c_byte = inp.read(1)
                if start == self.blocksize:
                    #print(self.curr_block)
                    encoded_data = utils.encode(args, net, self.curr_block)
                    self.write(encoded_data, out)
                    self.curr_block = np.zeros(self.blocksize, dtype=np.float32)
                    self.add_index()
                    start = self.index_size
                    end = start + 8
            #print("".join([str(int(i)) for i in self.curr_block]))

    def add_index(self):
        index_rem = self.index_size-1
        for pos in range(index_rem+1):
            cbit = self.index_counter >> index_rem & 1
            self.curr_block[pos] = cbit
            index_rem -= 1
        assert (index_rem == -1)
        self.index_counter += 1

    def insert_data(self, start, end, c_byte):
        rem_bits = 7
        for pos in range(start, end):
            self.curr_block[pos] = c_byte[0] >> rem_bits & 1
            rem_bits -= 1
        assert(rem_bits == -1)

    def write(self, encoded_data, outstream):
        outstream.write(">Fragment{};blocksize{};indexsize{};\n".format(self.index_counter, self.blocksize, self.index_size))
        outstream.write(encoded_data + "\n")


class BitDecodeStream(object):
    def __init__(self, inp_, outp_, blocksize, index_size):
        self.input = inp_
        self.output = outp_
        self.blocksize = blocksize
        self.index_size = index_size
        self.entries = dict()


    def read(self, args, net):
        with open(self.input, "r") as inp, open(self.output, "wb") as out:
            curr_index = 0
            for count, line in enumerate(inp, start=1):
                if count % 2 == 0:
                    line = line.rstrip("\n")
                    assert(all(b in "ACGT" for b in line))
                    decoded = utils.decode(args, net, line)
                    ind = int(decoded[:self.index_size], 2)
                    packet = decoded[self.index_size:]
                    padding = 0
                    if ind == 0:
                        packet, padding = self.remove_padding(packet)
                    if ind == curr_index:
                        self.write(packet, out, padding)
                        curr_index += 1
                        self.check_entries(curr_index, out)
                    else:
                        self.entries[str(ind)] = packet

    def check_entries(self, curr_index, out):
        ind = str(curr_index)
        if ind in self.entries.keys():
            self.write(self.entries[ind], out)
            del(self.entries[ind])
            curr_index += 1
            self.check_entries(curr_index, out)

    def remove_padding(self, block):
        pad = 0
        for i in range(0, self.blocksize, 8):
            if int(block[i:i+8], 2) == 0:
                pad += 8
            else:
                return block[i:], pad
        # special case for the first entry to remove all prepending zero bytes

    def write(self, decoded_data, outstream, padding=0):
        outstream.write(int(decoded_data, 2).to_bytes((self.blocksize - (self.index_size+padding))//8, 'big'))

if __name__ == '__main__':
    t = BitEncodeStream(inp_='../../../test_data/mosla.txt', outp_='../../../test_data/mosla_encoded.fasta',
                        blocksize=64, index_size=16)
    #t = BitDecodeStream(inp_='../../../test_data/mosla_encoded.fasta', outp_='../../../test_data/mosla_decoded.txt', blocksize=64, index_size=16)
    t.read(0, 0)
