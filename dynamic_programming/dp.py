# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 19:20:42 2017

@author: wattai
"""
# 学習データセットを作成するために，様々なサイトから楽曲情報を取得した際の,
# 曲名の表記ゆれを統一するスクリプト．

# FFMLS とは, 文字列が入ったリスト: X_list のそれぞれの文字列に,
# 最も似ている文字列を リスト: Y_list 中から探し出すアルゴリズム．

# Cython も書いたけど, さほど早くならなかった(速くて10%減)

import numpy as np
import pandas as pd
import multiprocessing as multi
from multiprocessing import Pool

import time


class FFMLS():

    def __init__(self, Y_list):

        self.Y_list = Y_list.copy()
        self.Y_left = Y_list.copy()
        self.X_left = []
        self.Z_match = []
        self.match_index = []
        self.unmatch_index = []

        self.likehood = []

    def len_matched_char_LCS_DP(self, X, Y):

        len_X = len(X)
        len_Y = len(Y)
        LCS = np.zeros([len_X+1, len_Y+1], dtype=np.int)

        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                d = 1 if x == y else 0
                LCS[i, j] = np.max(np.array(
                        [LCS[i-1, j-1]+d, LCS[i-1, j], LCS[i, j-1]]))

        return np.max(LCS) - np.abs(len_X - len_Y) * (Y.find(X) == -1)

    def fast_find_most_likely_strings(self):
        # X_list の各要素に最も似ている Y_list の各要素を割り当てる 鬼早い版
        result = []
        self.likehood = np.zeros([len(X_list), len(Y_list)]) - np.inf

        for i, X in enumerate(self.X_left):
            if X in self.Y_left:
                result.append(X)
                self.Y_left.remove(X)
            else:
                self.likehood[i, :len(self.Y_left)] = [
                        self.len_matched_char_LCS_DP(
                                X, Y) for Y in self.Y_left]
                y = self.Y_left[np.argmax(self.likehood[i, :])]
                result.append(y)
                self.Y_left.remove(y)
        print(self.likehood)

        return result

    def parallel_element(self, X):
        # fast_find_most_likely_strings_parallel の 並列化要素
        return self.Y_left[np.argmax(
                [self.len_matched_char_LCS_DP(X, Y) for Y in self.Y_left])]

    def fast_find_most_likely_strings_parallel(self):
        # X_list の各要素に最も似ている Y_list の各要素を割り当てる 並列(multiprocess)版
        p = Pool(multi.cpu_count())
        self.likehood = p.starmap(self.parallel_element, zip(self.X_left))
        p.close()
        return self.likehood

    def complete_match_ffmls(self, X_list):
        # 計算量削減のため, 前処理として完全一致している要素を調べる [最速(鬼より)]
        self.X_left = X_list.copy()

        for i, X in enumerate(X_list):
            if X in self.Y_left:
                self.Z_match.append(X)
                self.X_left.remove(X)
                self.Y_left.remove(X)
                self.match_index.append(i)
            else:
                self.unmatch_index.append(i)

        return np.array(self.Z_match + self.fast_find_most_likely_strings())[
                np.argsort(self.match_index + self.unmatch_index)].tolist()

    def complete_match_ffmls_parallel(self, X_list):
        # 計算量削減のため, 前処理として完全一致している要素を調べる [ちょっぱや(最速より鬼より)]
        self.X_left = X_list.copy()

        for i, X in enumerate(X_list):
            if (X in self.Y_left):
                self.Z_match.append(X)
                self.X_left.remove(X)
                self.Y_left.remove(X)
                self.match_index.append(i)
            else:
                self.unmatch_index.append(i)

        return np.array(
                self.Z_match + self.fast_find_most_likely_strings_parallel())[
                np.argsort(self.match_index + self.unmatch_index)].tolist()


def encode_LCS_DP(X, Y):

    LCS = np.zeros([len(X)+1, len(Y)+1], dtype=np.int)

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            d = 1 if x == y else 0
            LCS[i, j] = np.max(np.array(
                    [LCS[i-1, j-1]+d, LCS[i-1, j], LCS[i, j-1]]))

    return LCS[:-1, :-1]


def len_matched_char_LCS_DP(X, Y):

    len_X = len(X)
    len_Y = len(Y)
    LCS = np.zeros([len_X+1, len_Y+1], dtype=np.int)

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            d = 1 if x == y else 0
            LCS[i, j] = np.max(np.array(
                    [LCS[i-1, j-1]+d, LCS[i-1, j], LCS[i, j-1]]))

    return np.max(LCS) - np.abs(len_X - len_Y) * (Y.find(X) == -1)


def decode_LCS_DP(X, Y, LCS):

    result = []
    i, j = LCS.shape[0]-1, LCS.shape[1]-1

    while(LCS[i, j] > 0 and i >= 0 and j >= 0):

        while(LCS[i, j] == LCS[i-1, j] and i > 0):
            i -= 1

        while(LCS[i, j] == LCS[i, j-1] and j > 0):
            j -= 1

        if(LCS[i, j] > 0):
            result.append(X[np.argmax(LCS[:, j])])
            i -= 1
            j -= 1

    return result[::-1]


def solve_LCS_DP(X, Y):
    return decode_LCS_DP(X, Y, encode_LCS_DP(X, Y))


def find_most_likely_strings_parallel(X_list, Y_list):
    # X_list の各要素に最も似ている Y_list の各要素を割り当てる
    from joblib import Parallel, delayed
    likehood = np.array(Parallel(n_jobs=-1)(
            [delayed(len_matched_char_LCS_DP)(
                    X, Y) for Y in Y_list for X in X_list])
    ).reshape(len(X_list), len(Y_list))
    print(likehood)
    return np.array(Y_list)[np.argmax(likehood, axis=1)]


def find_most_likely_strings(X_list, Y_list):
    # X_list の各要素に最も似ている Y_list の各要素を割り当てる
    likehood = np.zeros([len(X_list), len(Y_list)]) - np.inf
    for i, X in enumerate(X_list):
        for j, Y in enumerate(Y_list):
            likehood[i, j] = len_matched_char_LCS_DP(X, Y)
    print(likehood)
    return np.array(Y_list)[np.argmax(likehood, axis=1)]


def parallel_element(X, Y_left):
    # fast_find_most_likely_strings_parallel の 並列化要素
    if X in Y_left:
        return X
    else:
        y = Y_left[np.argmax([len_matched_char_LCS_DP(X, Y) for Y in Y_left])]
        return y


def fast_find_most_likely_strings_parallel(X_list, Y_list):
    # X_list の各要素に最も似ている Y_list の各要素を割り当てる 並列(joblib)版
    from joblib import Parallel, delayed
    result = Parallel(n_jobs=-1)(
            [delayed(parallel_element)(X, Y_list) for X in np.array(X_list)])
    return result


def fast_find_most_likely_strings2(X_list, Y_list):
    # X_list の各要素に最も似ている Y_list の各要素を割り当てる 鬼早い版
    result = np.array([])
    likehood = np.zeros([len(X_list), len(Y_list)]) - np.inf
    Y_left = np.array(Y_list)

    # from joblib import Parallel, delayed
    for i, X in enumerate(np.array(X_list)):
        if X in Y_left:
            result = np.append(result, X)
            Y_left = np.delete(Y_left, np.where(Y_left == X)[0][0], axis=0)
        else:
            # for j, Y in enumerate(Y_left):
            #    likehood[i, j] = len_matched_char_LCS_DP(X, Y)

            likehood[i, :len(Y_left)] = [
                    len_matched_char_LCS_DP(X, Y) for Y in Y_left]
            y = Y_left[np.argmax(likehood[i, :])]
            result = np.append(result, y)
            Y_left = np.delete(Y_left, np.where(Y_left == y)[0][0], axis=0)

    print(likehood)
    return result.tolist()


def fast_find_most_likely_strings(X_list, Y_list):
    # X_list の各要素に最も似ている Y_list の各要素を割り当てる 鬼早い版
    result = []
    likehood = np.zeros([len(X_list), len(Y_list)]) - np.inf
    Y_left = Y_list.copy()

    for i, X in enumerate(X_list):
        if X in Y_left:
            result.append(X)
            Y_left.remove(X)
        else:
            likehood[i, :len(Y_left)] = [
                    len_matched_char_LCS_DP(X, Y) for Y in Y_left]
            y = Y_left[np.argmax(likehood[i, :])]
            result.append(y)
            Y_left.remove(y)
    print(likehood)
    return result


def complete_match_ffmls(X_list, Y_list):
    # 計算量削減のため, 前処理として完全一致している要素を調べる [最速(鬼より)]

    Z_match = []  # 完全一致した list
    X_left = X_list.copy()  # X_list の残り list
    Y_left = Y_list.copy()  # Y_list の残り list
    match_index = []  # 完全一致した index
    unmatch_index = []  # 完全一致しなかった index

    for i, X in enumerate(X_list):
        if X in Y_left:
            Z_match.append(X)
            X_left.remove(X)
            Y_left.remove(X)
            match_index.append(i)
        else:
            unmatch_index.append(i)

    return np.array(Z_match + fast_find_most_likely_strings(X_left, Y_left))[
            np.argsort(match_index + unmatch_index)].tolist()


if __name__ == '__main__':

    # X = 'abcbdab'
    # Y = 'bdcaba'

    # X = 'My opinions are similar to his.'
    # Y = 'Those have similar shapes.'

    X = 'www.python-izm.com'
    Y = 'www,python-izm,com'
    print(''.join(solve_LCS_DP(X, Y)))

    X = 'まりものおまもり'
    Y = 'もりのおまわりさん'
    print(''.join(solve_LCS_DP(X, Y)))

    # X = '†渚の小悪魔ラヴリィ〜レイディオ†(IIDX EDIT)'
    # Y = '†渚の小悪魔ラヴリィ~レイディオ†(IIDX EDIT)'

    # X = 'AA'
    # Y = 'A'

    X_list = 1 * ['Aegis', 'ALBIDA', 'Answer', 'Anthem Landing',
                  'Believe in Me', 'BLUST oF WIND', 'Bounce Bounce Bounce',
                  'Breaking the ground', 'Broken', 'Broken Eden',
                  'Dances with Snow Fairies']
    Y_list = 1 * ['Aegis', 'ALBIDA', 'Answer', 'ANTHEM LANDING',
                  'Believe In ♡ Me', 'BLUST OF WIND', 'Bounce Bounce Bounce',
                  'Breaking the ground', 'Broken', 'BROKEN EDEN',
                  'Dances with Snow Fairies']

    x_sp11 = pd.read_csv('x11_names.csv', encoding="sjis", delimiter=',',
                         ).values.squeeze().tolist()
    y_sp11 = pd.read_csv('y11_names.csv', encoding="sjis", delimiter=',',
                         ).values.squeeze().tolist()
    x_sp12 = pd.read_csv('x12_names.csv', encoding="sjis", delimiter=',',
                         ).values.squeeze().tolist()
    y_sp12 = pd.read_csv('y12_names.csv', encoding="sjis", delimiter=',',
                         ).values.squeeze().tolist()

    X_list = sorted(x_sp12 + x_sp11)
    Y_list = sorted(y_sp12 + y_sp11)

    s = time.time()
    # result1 = find_most_likely_strings_parallel(X_list, Y_list) # 遅すぎ注意
    # print(result1)
    print('time to process1: %.2f [sec]' % (time.time() - s))

    s = time.time()
    # result2 = complete_match_ffmls(X_list, Y_list)
    # print(result2)
    print('time to process2: %.2f [sec]' % (time.time() - s))

    s = time.time()
    # result3 = fast_find_most_likely_strings(X_list, Y_list)
    # print(result3)
    print('time to process3: %.2f [sec]' % (time.time() - s))

    s = time.time()
    # result4 = fast_find_most_likely_strings_parallel(X_list, Y_list
    #                                                 ) # Evans だけ間違える
    # print(result3)
    print('time to process4: %.2f [sec]' % (time.time() - s))

    s = time.time()
    f = FFMLS(Y_list)
    result5 = f.complete_match_ffmls(X_list)
    # print(result5)
    print('time to process5: %.2f [sec]' % (time.time() - s))

    s = time.time()
    f = FFMLS(Y_list)
    result6 = f.complete_match_ffmls_parallel(X_list)
    # print(result6)
    print('time to process6: %.2f [sec]' % (time.time() - s))

    # print('check for output result: ', result1.sort() == result2.sort() )
    print('length of X_list: %d' % len(X_list))
    print('length of Y_list: %d' % len(Y_list))

    # print(X_list)
