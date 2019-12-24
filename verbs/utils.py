def get_idx(word):
    idx_list = []
    for i in range(len(word)):
        idx_list.append(ord(word[i]) - 97)
    return idx_list

def get_idx_add_eol(word):
    idx_list = get_idx(word)
    idx_list.append(27)
    return idx_list