import torch
from models.modules.bit_pruning import count_bit, truncation, bit_sparse

int_number = torch.ones(256)
for i in range(256):
    int_number[i] = i - 128
print(int_number)
bit_cnt = count_bit(int_number)
print('bit_cnt: \n{}'.format(bit_cnt))
int_number_index = bit_cnt < 3
print('int_number_index:\n{}'.format(int_number_index))

code = []
for i in range(256):
    if int_number_index[i] > 0:
        code.append(int(int_number[i].item()))
print(code)

# [-128, -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]

# [-128, -96, -80, -72, -68, -66, -65, -64, -48, -40, -36, -34, -33, -32, -24, -20, -18, -17, -16, -12, -10,
# -9, -8, -6, -5, -4, -3, -2, -1,
# 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 17, 18, 20, 24, 32, 33, 34, 36, 40, 48, 64, 65, 66, 68, 72, 80, 96]


# [-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]

# [-128, -96, -80, -72, -68, -66, -65, -64, -48, -40, -36, -34, -33, -32, -24, -20, -18, -17, -16, -12, -10,
# -9, -8, -6, -5, -4, -3, -2, -1,
# 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 17, 18, 20, 24, 32, 33, 34, 36, 40, 48, 64, 65, 66, 68, 72, 80, 96]
