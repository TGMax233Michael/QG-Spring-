def quick_sort(seq: list):
    if len(seq) <= 1:
        return seq

    pivot = seq[-1]

    greater = [item for item in seq if item > pivot]
    mid = [item for item in seq if item == pivot]
    smaller = [item for item in seq if item < pivot]

    return quick_sort(smaller) + mid + quick_sort(greater)

seq01 = [1, 3, 5, 6, 7, 2, 4, 1]
print(quick_sort(seq01))