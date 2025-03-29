def count_sort(seq: list, max_value):
    count_list = [0 for i in range(max_value+1)]
    for item in seq:
        count_list[item] += 1

    j = 0
    for i in range(len(count_list)):
        while count_list[i] > 0:
            seq[j] = i
            count_list[i] -= 1
            j += 1

    return seq

seq = [3, 4, 0, 1, 4,8]
print(count_sort(seq, max(seq)))