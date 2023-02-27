import numpy as np

ByteIndexMask64 = np.ndarray(8,dtype=np.uint64)
ByteIndexMask64[0] = np.uint64(0xff00000000000000)
for i in range(1,8):
    ByteIndexMask64[i] = ByteIndexMask64[i-1] 
    ByteIndexMask64[i] >>= np.uint64(8)
def byteindex64(val,bindex):
    val = np.uint64(val) & ByteIndexMask64[bindex]
    bindex_rev = 7-bindex
    val >>= np.uint64(bindex_rev*8)
    return(val)

# from https://stackoverflow.com/questions/63954102/numpy-vectorized-way-to-count-non-zero-bits-in-array-of-integers
def bit_count(arr):
     # Make the values type-agnostic (as long as it's integers)
     t = arr.dtype.type
     mask = t(-1)
     if arr.dtype == np.uint64:
          s55 = t(0x5555555555555555) & mask  # Add more digits for 128bit support
          s33 = t(0x3333333333333333) & mask
          s0F = t(0x0F0F0F0F0F0F0F0F) & mask
          s01 = t(0x0101010101010101) & mask
     else:
          s55 = t(0x5555555555555555 & mask) 
          s33 = t(0x3333333333333333 & mask) 
          s0F = t(0x0F0F0F0F0F0F0F0F & mask) 
          s01 = t(0x0101010101010101 & mask) 

     arr = arr - ((arr >> t(1)) & s55)
     arr = (arr & s33) + ((arr >> t(2)) & s33)
     arr = (arr + (arr >> t(4))) & s0F
     counts =  (arr * s01) >> t(8 * (arr.itemsize - 1))
     return(counts.sum())

class guint8(np.uint8):
    def __new__(cls,val):
        gu = np.uint8.__new__(cls,val)
        return(gu)
    
    def __getitem__(self,idx):
        if idx >= 8 or idx < 0:
            raise IndexError("Index out of range for bit-references on a guint8")
        try:
            idx_mask = np.uint8(1 << idx)
        except:
            raise TypeError("Index must be an integer")
        out = np.bool8((self & idx_mask) > 0)
        return(out)
    
    def __setitem__(self,idx,val):
        raise NotImplemented("guint8 is immutable")
    

class Gbytes_Iter:
    def __init__(self,gb):
        self.gb = gb
        self._iter_index = 0
    
    def __next__(self):
        if self._iter_index >= self.gb.num_bytes:
            raise StopIteration
        ret = self.gb[self._iter_index]
        self._iter_index += 1
        return(ret)

class GBytes:
    def __init__(self, b):
        num_bytes = len(b)
        self.num_bytes = num_bytes
        self.num_qwords = num_bytes//8
        self.num_qword_bytes = self.num_qwords*8
        self.num_rem = num_bytes - self.num_qword_bytes

        self.qword_aligned = False
        self.qword_array = np.frombuffer(b[:self.num_qword_bytes],dtype=np.dtype('>u8')) 
        self.rem_array = np.frombuffer(b[self.num_qword_bytes:],dtype=np.uint8) 

    def _get_byte(self,idx):
        if idx < self.num_qword_bytes:
            qword_idx = idx // 8
            byte_idx = idx % 8
            qword = self.qword_array[qword_idx]
            byte = byteindex64(qword,byte_idx) 
        elif idx < self.num_bytes:
            rem_idx = idx - self.num_qword_bytes
            byte = self.rem_array[rem_idx]
        else:
            raise IndexError("Index out-of-range")
        return(byte)
            
    def __len__(self):
        return(self.num_bytes)

    def __getitem__(self,idx):
        if isinstance(idx,slice):
            start = idx.start
            if start == None:
                start = 0
            stop = idx.stop
            if stop == None:
                stop = self.num_bytes
            size = stop - start
            tmp_array = np.ndarray(size,dtype=np.uint8)
            for i,j in enumerate(range(start,stop)):
                tmp_array[i] = self._get_byte(j) 
            out = GBytes(tmp_array)
            return(out)
        byte = guint8(self._get_byte(idx))
        return(byte)
    
    def _very_shallow_cpy(self):
        new = self.__class__.__new__(self.__class__)
        new.num_bytes = self.num_bytes 
        new.num_qwords = self.num_qwords 
        new.num_qword_bytes = self.num_qword_bytes 
        new.num_rem = self.num_rem 
        return(new)
    
    @staticmethod
    def bitwise_xor(gb1,gb2):
        qword_array = np.bitwise_xor(gb1.qword_array,gb2.qword_array)
        rem_array = np.bitwise_xor(gb1.rem_array,gb2.rem_array)
        new = gb1._very_shallow_cpy()
        new.qword_array = qword_array
        new.rem_array = rem_array
        return(new)
    
    @staticmethod
    def bitwise_and(gb1, gb2):
        qword_array = np.bitwise_and(gb1.qword_array,gb2.qword_array)
        rem_array = np.bitwise_and(gb1.rem_array,gb2.rem_array)
        new = gb1._very_shallow_cpy()
        new.qword_array = qword_array
        new.rem_array = rem_array
        return(new)
    
    @staticmethod
    def bitwise_or(gb1, gb2):
        qword_array = np.bitwise_or(gb1.qword_array,gb2.qword_array)
        rem_array = np.bitwise_or(gb1.rem_array,gb2.rem_array)
        new = gb1._very_shallow_cpy()
        new.qword_array = qword_array
        new.rem_array = rem_array
        return(new)
    
    @staticmethod
    def bitwise_not(gb1):
        qword_array = np.bitwise_not(gb1.qword_array)
        rem_array = np.bitwise_not(gb1.rem_array)
        new = gb1._very_shallow_cpy()
        new.qword_array = qword_array
        new.rem_array = rem_array
        return(new)
    
    @staticmethod
    def bitwise_not(gb1):
        qword_array = np.bitwise_not(gb1.qword_array)
        rem_array = np.bitwise_not(gb1.rem_array)
        new = gb1._very_shallow_cpy()
        new.qword_array = qword_array
        new.rem_array = rem_array
        return(new)
        
    def bit_count(self):
        qword_bitcount = bit_count(self.qword_array)
        rem_bitcount = bit_count(self.rem_array)
        count = qword_bitcount + rem_bitcount
        return(count)

    @staticmethod
    def similarity(gb1, gb2):
        qword = np.bitwise_not( np.bitwise_xor( gb1.qword_array,gb2.qword_array ) )
        rem = np.bitwise_not( np.bitwise_xor( gb1.rem_array,gb2.rem_array ) )
        count =  bit_count(qword)
        count += bit_count(rem)
        return(count)

    def __iter__(self):
        return(Gbytes_Iter(self))

    def __repr__(self):
        s = 'g\''
        for b in self:
            s += '\\x{:02x}'.format(b)
        s += '\''
        return(s)

