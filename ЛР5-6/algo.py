if __name__ == '__main__':
    n_m = input()
    lst_1 = [int(i) for i in n_m.split()]
    n, m = lst_1[0], lst_1[1]

    a_i = input()
    arr = [int(i) for i in a_i.split()]

    intervals = []
    for i in range(n):
        lst_i = input()
        intervals.append([int(i) for i in lst_i.split()])

    occurrences = [0] * n
    for i in intervals:
        for j in range(i[0] - 1, i[1] - 1):
            occurrences[j] += 1

    result = 0

    for i in intervals:
        max_ = 0
        max_index = 0
        for j in range(i[0] - 1, i[1] - 1):
            if max_ < occurrences[j]:
                max_ = occurrences[j]
                max_index = j
        result = result | occurrences[max_index]
    print(result)
