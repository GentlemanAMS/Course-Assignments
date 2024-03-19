filename = "EE19B141_EE19B001.txt"

def bitstring_to_bytes(s):
    v = int(s, 2)
    b = []
    while v:
        b.append(int(v & 0xff))
        v >>= 8
    #print b
    return b[::-1]

def readfile(filename):
    ct_list = []
    fct_list = []
    print("Read the contents of the File")
    fn= open(filename,'r')
    cont = fn.readlines()
    type(cont) 

    for i in range(1,len(cont)):
        data = cont[i].split('\t')
        data[1] = data[1][:-1]
        ct1 = bitstring_to_bytes(bin(int(data[0], 16)))
        ct_list.append(ct1)
        fct_t = data[1].replace('\n','')
        fct = bitstring_to_bytes(bin(int(fct_t, 16)))
        fct_list.append(fct)
        
    return ct_list,fct_list


def convert_to_multidim_array(text):
    multidim_array = [[],[],[],[]]
    for i in range(16):
        multidim_array[int(i/4)].append(text[i])
    return multidim_array

def shift_rows(s):
    s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
    s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
    s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]
    return s

xtime = lambda a: (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)

def mix_single_column(a):
    t = a[0] ^ a[1] ^ a[2] ^ a[3]
    u = a[0]
    a[0] ^= t ^ xtime(a[0] ^ a[1])
    a[1] ^= t ^ xtime(a[1] ^ a[2])
    a[2] ^= t ^ xtime(a[2] ^ a[3])
    a[3] ^= t ^ xtime(a[3] ^ u)


def mix_columns(s):
    for i in range(4):
        mix_single_column(s[i])
    return s



isbox = [0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
         0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
         0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
         0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
         0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
         0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
         0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
         0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
         0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
         0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
         0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
         0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
         0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
         0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
         0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
         0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d]

###########################################################################################################################################################

# def recursion_function(cipher_text, faulty_cipher_text, fault_matrix, byte_no, key_candidates, result_valid):
    
#     iter_column = int(byte_no/4)
#     iter_row = byte_no%4
#     for keyguess in range(256):
#         temp = isbox[keyguess ^ cipher_text[iter_column][iter_row]] ^ isbox[keyguess ^ faulty_cipher_text[iter_column][iter_row]]
#         pass


#     result_valid = True
#     if byte_no == 16:
#         return key_candidates, result_valid
#     pass

# def find_candidate_key(cipher_text, faulty_cipher_text, fault_matrix):
    
#     key_candidates = []
#     return_candidates = []
#     result_valid = True

#     key_candidates, result_valid = recursion_function(cipher_text, faulty_cipher_text, fault_matrix, 0, key_candidates, result_valid)

#     for iter in range(16):
#         for iter_row in range(4):
#             for keyguess in range(256):
#                 temp = isbox[keyguess ^ cipher_text[iter_column][iter_row]] ^ isbox[keyguess ^ faulty_cipher_text[iter_column][iter_row]]
#                 # print(key_candidates[iter_column][iter_row])
#                 if temp == fault_matrix[iter_column][iter_row]:
#                     key_candidates[iter_column*4+iter_row].extend([keyguess])
#             if key_candidates[iter_column*4+iter_row] == []:
#                 result_valid = False
#                 return key_candidates, result_valid

#     for byte_no in range(16):
#         number_key_candidate_byte = len(key_candidates[byte_no])
#         temp = []
    
#     return key_candidates, result_valid

###########################################################################################################################################################

# def find_candidate_key(cipher_text, faulty_cipher_text, fault_matrix):
    
#     key_candidates = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
#     return_candidates = []
#     result_valid = True
#     for iter_column in range(4):
#         for iter_row in range(4):
#             for keyguess in range(256):
#                 temp = isbox[keyguess ^ cipher_text[iter_column][iter_row]] ^ isbox[keyguess ^ faulty_cipher_text[iter_column][iter_row]]
#                 # print(key_candidates[iter_column][iter_row])
#                 if temp == fault_matrix[iter_column][iter_row]:
#                     key_candidates[iter_column*4+iter_row].extend([keyguess])
#             if key_candidates[iter_column*4+iter_row] == []:
#                 result_valid = False
#                 return key_candidates, result_valid

#     for byte_no in range(16):
#         number_key_candidate_byte = len(key_candidates[byte_no])
#         temp = []
    
#     return key_candidates, result_valid

###########################################################################################################################################################
# def find_candidate_key(cipher_text, faulty_cipher_text, fault_matrix):
    
#     key_candidates = []

#     for keyguess0 in range(256):
#         key0 = isbox[keyguess0 ^ cipher_text[0][0]] ^ isbox[keyguess0 ^ faulty_cipher_text[0][0]]
#         if key0 == fault_matrix[0][0]:
#             for keyguess in range(256):
#                 key1 = isbox[keyguess ^ cipher_text[0][1]] ^ isbox[keyguess ^ faulty_cipher_text[0][1]]
#                 if key1 == fault_matrix[0][1]:
#                     for keyguess in range(256):
#                         key0 = isbox[keyguess ^ cipher_text[0][0]] ^ isbox[keyguess ^ faulty_cipher_text[0][0]]
#                         if key0 == fault_matrix[0][0]:
#                             for keyguess in range(256):
#                                 key1 = isbox[keyguess ^ cipher_text[0][1]] ^ isbox[keyguess ^ faulty_cipher_text[0][1]]
#                                 if key1 == fault_matrix[0][1]:
                
#     return key_candidates, result_valid

###########################################################################################################################################################



###########################################################################################################################################################

# def attack_8th_round(cipher_text, faulty_cipher_text, column_in_attack):
    
#     key_candidates = []

#     # for gamma1 in range(1,3):
#     #     for gamma2 in range(1,3):
#     #         for gamma3 in range(1,3):
#     #             for gamma4 in range(1,3):
#     for gamma1 in range(1,256):
#         for gamma2 in range(1,256):
#             for gamma3 in range(1,256):
#                 for gamma4 in range(1,256):
#                     gamma = [gamma1, gamma2, gamma3, gamma4]
#                     if column_in_attack == 0:
#                         fault_matrix = [gamma,[0,0,0,0],[0,0,0,0],[0,0,0,0]]
#                         fault_matrix = mix_columns(shift_rows(fault_matrix))    
#                     if column_in_attack == 1:
#                         fault_matrix = [[0,0,0,0],gamma,[0,0,0,0],[0,0,0,0]]
#                         fault_matrix = mix_columns(shift_rows(fault_matrix))    
#                     if column_in_attack == 2:
#                         fault_matrix = [[0,0,0,0],[0,0,0,0],gamma,[0,0,0,0]]
#                         fault_matrix = mix_columns(shift_rows(fault_matrix))    
#                     if column_in_attack == 3:
#                         fault_matrix = [[0,0,0,0],[0,0,0,0],[0,0,0,0],gamma]
#                         fault_matrix = mix_columns(shift_rows(fault_matrix))
#                     print(fault_matrix)    
#                     candidate, result_valid = find_candidate_key(cipher_text, faulty_cipher_text, fault_matrix)
#                     if result_valid == True:
#                         print(result_valid)
#                         key_candidates.append(candidate)
#                     else:
#                         continue

#     return key_candidates

###########################################################################################################################################################




key_candidates = []

def recursion_function(cipher_text, faulty_cipher_text, fault_matrix, keys_until_now, byte_no):
    
    if byte_no == 16:
        key_candidates.append(keys_until_now)
        return True
    
    iter_column = int(byte_no/4)
    iter_row = byte_no%4

    for keyguess in range(256):
        tempguess = isbox[keyguess ^ cipher_text[iter_column][iter_row]] ^ isbox[keyguess ^ faulty_cipher_text[iter_column][iter_row]]
        if tempguess == fault_matrix[iter_column][iter_row]:
            temp = keys_until_now.copy()
            temp.extend(tempguess)
            recursion_function(cipher_text, faulty_cipher_text, fault_matrix,temp, byte_no+1)
    
def find_candidate_key(cipher_text, faulty_cipher_text, fault_matrix):
    recursion_function(cipher_text, faulty_cipher_text, fault_matrix, [], 0)

def attack_8th_round(cipher_text, faulty_cipher_text, column_in_attack):
    # for gamma1 in range(1,3):
    #     for gamma2 in range(1,3):
    #         for gamma3 in range(1,3):
    #             for gamma4 in range(1,3):
    for gamma1 in range(1,256):
        for gamma2 in range(1,256):
            for gamma3 in range(1,256):
                for gamma4 in range(1,256):
                    gamma = [gamma1, gamma2, gamma3, gamma4]
                    if column_in_attack == 0:
                        fault_matrix = [gamma,[0,0,0,0],[0,0,0,0],[0,0,0,0]]
                        fault_matrix = mix_columns(shift_rows(fault_matrix))    
                    if column_in_attack == 1:
                        fault_matrix = [[0,0,0,0],gamma,[0,0,0,0],[0,0,0,0]]
                        fault_matrix = mix_columns(shift_rows(fault_matrix))    
                    if column_in_attack == 2:
                        fault_matrix = [[0,0,0,0],[0,0,0,0],gamma,[0,0,0,0]]
                        fault_matrix = mix_columns(shift_rows(fault_matrix))    
                    if column_in_attack == 3:
                        fault_matrix = [[0,0,0,0],[0,0,0,0],[0,0,0,0],gamma]
                        fault_matrix = mix_columns(shift_rows(fault_matrix))

                    find_candidate_key(cipher_text, faulty_cipher_text, fault_matrix)

    return key_candidates





def find_valid_candidates(key_candidates):
    pass


ct_list, fct_list = readfile(filename)

#for column_on_attack in range(1):
for column_on_attack in range(4):
    key_candidates = []
    #for iter in range(4):
    for iter in range(len(ct_list)):
        cipher_text = convert_to_multidim_array(ct_list[iter])
        faulty_cipher_text = convert_to_multidim_array(fct_list[iter])
        result = attack_8th_round(cipher_text, faulty_cipher_text, column_on_attack)
        #print(result)
        key_candidates.append(result)
    
    valid_candidates = find_valid_candidates(key_candidates)







