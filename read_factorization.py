import numpy as np
from sympy import *
import sys

try:
    data = np.load('factorizations_f2.npz')
except FileNotFoundError:
    raise FileNotFoundError('Get factorizations_f2.npz from https://github.com/google-deepmind/alphatensor/blob/1949163da3bef7e3eb268a3ac015fd1c2dbfc767/algorithms/factorizations_f2.npz first')

# run with e.g python3 read_factorization.py 2,2,2
factors = data[sys.argv[1]]

# https://github.com/google-deepmind/alphatensor/blob/1949163da3bef7e3eb268a3ac015fd1c2dbfc767/benchmarking/utils.py#L57
assert factors[0].shape[0] == factors[1].shape[0]
assert factors[1].shape[0] == factors[2].shape[0]
factors = [factors[0].copy(), factors[1].copy(), factors[2].copy()]
n = int(np.sqrt(factors[0].shape[0]))
rank = factors[0].shape[-1]
factors[0] = factors[0].reshape(n, n, rank)
factors[1] = factors[1].reshape(n, n, rank)
factors[2] = factors[2].reshape(n, n, rank)
# The factors are for the transposed (symmetrized) matrix multiplication
# tensor. So to use the factors, we need to transpose back.
factors[2] = factors[2].transpose(1, 0, 2)

for alpha in range(rank):
    terms = []
    for i in range(n):
        for j in range(n):
            if factors[0][i, j, alpha] == 0: continue
            terms.append((i, j))
    if len(terms) == 1 and alpha != 0:
        print('tmp_mk.clear();')

    if len(terms) >= 2:
        print(f'tmp_mk.set_to_sum_unchecked(&a_blks[{terms[0][0]}][{terms[0][1]}], &a_blks[{terms[1][0]}][{terms[1][1]}]);')
        terms = terms[2:]

    for i, j in terms:
        print(f'tmp_mk.add_unchecked(&a_blks[{i}][{j}]);')


    terms = []
    for j in range(n):
        for k in range(n):
            if factors[1][j, k, alpha] == 0: continue
            terms.append((j, k))

    if len(terms) == 1 and alpha != 0:
        print('tmp_kn.clear();')

    if len(terms) >= 2:
        print(f'tmp_kn.set_to_sum_unchecked(&b_blks[{terms[0][0]}][{terms[0][1]}], &b_blks[{terms[1][0]}][{terms[1][1]}]);')
        terms = terms[2:]

    for j, k in terms:
        print(f'tmp_kn.add_unchecked(&b_blks[{j}][{k}]);')

    if alpha != 0:
        print('tmp_mn.clear();')

    print(f'addmul_recurse(&mut tmp_mn, &tmp_mk, &tmp_kn, &algos);')

    for i in range(n):
        for k in range(n):
            if factors[2][i, k, alpha] == 0: continue
            print(f'c_blks[{i}][{k}].add_unchecked(&tmp_mn);')