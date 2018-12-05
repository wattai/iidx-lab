#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:35:33 2017

@author: wattai
"""

# 使う前に Chrome Driver を以下からダウンロードして,このスクリプトと同じディレクトリに置いてね．
# http://chromedriver.chromium.org/downloads
# Driver でエラーが出たら,大体 Driver を新しいのに変えれば直るよ．

# 解決済 ### Lv.1 などで1つもノーツが降らないレーンがあると要素数が変わり,誤動作する ###
# 解決済 ### 大犬のワルツ,スクラッチ来ないので誤動作する ###
### 最新作だと,低レベル帯にtextageがまだ譜面を載せていない場合がある ###
# 解決済 ### TODO: numpy配列にdtypeと列ごとの名前付ける, ヘッダ書く ###
### Firefox だと DAY DREAM でgraphの表示が文字化けする ###
### Chrome だと 画面内に keyinfo のボタンがないと
### alert の表示が止まるときがある. 'k' を押せば続行し,
### 画面内に必ずボタンが表示されるように window を縦に伸ばせば, 続行できる ###
# 解決済 1小節内の最大鍵盤密度と皿密度とノーツ密度に,それぞれBPMをかけた数値を抽出する ###
# 解決済 bpm / 4 / 60 * note_per_bar ### <- note / second
# 解決済 秒間ノーツ密度を DataFrame に追加する ###
### CN, BSS中に同時に降る秒間ノーツ密度 を抽出する ###
### 各小節での各レーンのみでのノーツ密度を計算して 秒間最大縦連ノーツ密度 を抽出する ###
# 解決済 z のソート結果と result2 のソート結果が異なる (表記ゆれがあるので当たり前感) ###
# 解決済 全部 LCS で直接解く必要があるか (complete match を使わず) <- LCSを解く直前で complete match をチェックし, LCSの呼び出し回数を削減 ###

import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
lamp_list = ['FAILED', 'ASSIST CLEAR', 'EASY CLEAR', 'CLEAR', 'HARD CLEAR', 'EX HARD CLEAR', 'FULLCOMBO CLEAR']
le.fit(lamp_list)

dic_trans = {'FAILED':0, 'ASSIST CLEAR':1, 'EASY CLEAR':2, 'CLEAR':3, 'HARD CLEAR':4, 'EX HARD CLEAR':5, 'FULLCOMBO CLEAR':6}
dic_inver = {0:'FAILED', 1:'ASSIST CLEAR', 2:'EASY CLEAR', 3:'CLEAR', 4:'HARD CLEAR', 5:'EX HARD CLEAR', 6:'FULLCOMBO CLEAR'} 


from matplotlib.colors import LinearSegmentedColormap
def generate_cmap(colors):
    """自分で定義したカラーマップを返す"""
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)


def scrape_textage(mode='s', difficulty=12, fname='test_music_data.csv'):
    # mode: 's' or 'd'. 's' means SP, 'd' means DP.
    # difficulty: iidx-score difficulty.
    # fname: filename of scraped csv data.
    # n_limit: upper bound of scraping time number.
    
    difficulty_hex = hex(difficulty)[2:].upper()    

    I = np.zeros([0, 33])
    
    delay = 10 # out of time for scraping.
    
    url = "http://textage.cc/score/?%s%s01B10"%(mode, difficulty_hex) # ?s1... => SP lv.1, ?dC... => DP lv.12
    
    #browser = webdriver.Firefox()
    browser = webdriver.Chrome()#'./chromedriver')
    browser.get(url)
    
    table_title = browser.find_element_by_xpath('.//tr[1]/th[4]').text
    n_score = int(table_title.split(' ')[-2].split('[')[1].split(']')[0])
    
    for l in range(2, n_score+2):
        try:
            xpath = "/html/body/center/table[2]/tbody/tr[%d]/td[2]/a[1]" %l
            
            element = WebDriverWait(browser, delay).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
            element.click()                              
            #browser.find_element_by_xpath(xpath).click()
            
            print("ok")
            element = WebDriverWait(browser, delay).until(
                EC.presence_of_element_located((By.XPATH, './/nobr/b'))
            )
            music_title = browser.find_element_by_xpath('.//nobr/b').text
            print("ok2")
            element = WebDriverWait(browser, delay).until(
                EC.presence_of_element_located((By.XPATH, './/nobr/font'))
            )
            music_type = browser.find_element_by_xpath('.//nobr/font').text
            print("ok3")
            element = WebDriverWait(browser, delay).until(
                EC.presence_of_element_located((By.XPATH, './/nobr'))
            )
            music_details = browser.find_element_by_xpath('.//nobr').text
    
 
            music_details = np.array(music_details.split())                  
            music_bpms = music_details[len(music_details) - 4][4:].split(u"～") # 謎のチルダで分割
            
    
            soup = BeautifulSoup(browser.page_source, 'html.parser')
            n_sofran = np.array([str(soup).count('t.gif')])
            if n_sofran > 0: n_sofran -= 1           



            #/html/body/table/tbody/tr/td[2]/table[1]/tbody/tr/td/div
            #/html/body/table/tbody/tr/td[2]/table[2]/tbody/tr/td/div
            #/html/body/table/tbody/tr/td[2]/table[4]/tbody/tr/td/div
            #/html/body/table/tbody/tr/td[19]/table[5]

            # 秒間ノーツ密度計算 ------------------------------------------
            if len(music_bpms) == 1:
                now_bpm = int(music_bpms[0])
                
            bw_notes_per_second = np.zeros([0])
            s_notes_per_second = np.zeros([0])
            both_notes_per_second = np.zeros([0])
            
            import re
            soup = BeautifulSoup(browser.page_source, 'html.parser')
            td = soup.select('body > table > tbody > tr > td')
            n_column = len(td)
            #note_list = []
            for i in range(0, n_column, 1):
                table = td[i].select('table')
                n_bar_in_column = len(table)
                for j in range(n_bar_in_column-1, -1, -1):
                    img = table[j].select('tbody > tr > td > div > img')
                    span = table[j].select('tbody > tr > td > div > span')
                    cnt_sofran = 0
                    cnt_bw_note = 0
                    cnt_s_note = 0
                    loc_sofran = np.zeros([0])
                    loc_bw_note = np.zeros([0])
                    loc_s_note = np.zeros([0])
                    now_bpms = np.zeros([0])
                    for k in range(0, len(img), 1):
                        if img[k].get('src') == '../t.gif':
                            cnt_sofran += 1
                            loc_sofran = np.concatenate((loc_sofran, np.array([img[k].get('style').split(':')[1][:-2]]).astype('int')), axis=0)
                            #print(loc_sofran)
                            now_bpms = np.concatenate((now_bpms, np.array(span[k].text.split()).astype('int')), axis=0)
                            #print(now_bpms)
                        elif img[k].get('src') == '../w.gif' or  img[k].get('src') == '../b.gif':
                            cnt_bw_note += 1
                            loc_bw_note = np.concatenate((loc_bw_note, np.array([re.split('[:;]', img[k].get('style'))[1][:-2]]).astype('int')), axis=0)
                        elif img[k].get('src') == '../s.gif':
                            cnt_s_note += 1
                            loc_s_note = np.concatenate((loc_s_note, np.array([re.split('[:;]', img[k].get('style'))[1][:-2]]).astype('int')), axis=0)
                    
                    
                    loc_bw_note.sort()
                    loc_s_note.sort()
                    bw_note_per_second = 0
                    s_note_per_second = 0
                    both_note_per_second = 0
                    if cnt_sofran > 0:
                        if (len(loc_bw_note[loc_bw_note >= loc_sofran[-1]]) + len(loc_s_note[loc_s_note >= loc_sofran[-1]])) <= 0:
                            now_bpm = now_bpms[-1]
                        else:
                            bw_note_per_second += now_bpm / 4 / 60 * len(loc_bw_note[loc_bw_note >= loc_sofran[-1]])
                            s_note_per_second += now_bpm / 4 / 60 * len(loc_s_note[loc_s_note >= loc_sofran[-1]])
                        #print(now_bpm)
                        #print(now_bpms)
                        #print(loc_sofran)
                        
                        for k in range(0, cnt_sofran, 1):
                            flag_above_bw_note = loc_bw_note <= loc_sofran[k]
                            flag_above_s_note = loc_s_note <= loc_sofran[k]
                            
                            bw_note_per_second += now_bpms[k] / 4 / 60 * len(loc_bw_note[flag_above_bw_note])
                            s_note_per_second += now_bpms[k] / 4 / 60 * len(loc_s_note[flag_above_s_note])

                            loc_bw_note = loc_bw_note[~flag_above_bw_note]
                            loc_s_note = loc_s_note[~flag_above_s_note]
                            
                        now_bpm = now_bpms[0]
                    else:
                        #print(now_bpm)
                        bw_note_per_second =  now_bpm / 4 / 60 * cnt_bw_note
                        s_note_per_second = now_bpm / 4 / 60 * cnt_s_note
                    
                    both_note_per_second = bw_note_per_second + s_note_per_second
                    
                    bw_notes_per_second = np.concatenate((bw_notes_per_second, np.array([bw_note_per_second])), axis=0)
                    s_notes_per_second = np.concatenate((s_notes_per_second, np.array([s_note_per_second])), axis=0)
                    both_notes_per_second = np.concatenate((both_notes_per_second, np.array([both_note_per_second])), axis=0)
                    
            #print(both_notes_per_second)
            #print(both_notes_per_second.tolist())
            try:
                print(both_notes_per_second[132])
                print(both_notes_per_second[152])
            except:
                print('')
            # 秒間ノーツ密度の系列が得られる
            print('max_both: ', np.max(both_notes_per_second))
            print('max_bw: ', np.max(bw_notes_per_second))
            print('max_s: ', np.max(s_notes_per_second))
            # ------------------------------------------------------
            
            element = WebDriverWait(browser, delay).until(
                EC.presence_of_element_located((By.XPATH, '/html/body'))
            )
            element.send_keys('k')
            #browser.find_element_by_xpath('/html/body/input[11]').click()
            alert = WebDriverWait(browser, delay).until(EC.alert_is_present(),
                                           'Timed out waiting for PA creation ' +
                                           'confirmation popup to appear.')
            #alert = browser.switch_to_alert()
            
            keyinfo = alert.text
            #time.sleep(1)
            alert.accept()


            element = WebDriverWait(browser, delay).until(
                EC.presence_of_element_located((By.XPATH, '/html/body'))
            )
            element.send_keys('g')
            #browser.find_element_by_xpath('/html/body/input[12]').click()
            alert = WebDriverWait(browser, delay).until(EC.alert_is_present(),
                                           'Timed out waiting for PA creation ' +
                                           'confirmation popup to appear.')
            #alert = browser.switch_to_alert()
            graph = alert.text
            alert.accept()
            
            #browser.close()
            browser.back()
            #browser.get(url)
            
            print("alert accepted")
        
            
        
            # 細かい抽出処理など
            #time.sleep(1)
            keyinfo = np.array(keyinfo.replace('l', ' ').split())
            graph = np.array(graph.split())
            music_title = np.array([music_title])
            music_type = np.array([music_type[1:-1]])
            
            play_style = np.array([(music_type[0].split())[0]])
            music_mode = np.array([(music_type[0].split())[1]])
    
            
            music_difficulty = np.array([music_details[len(music_details) - 2][1:]])#.astype(np.int)
            music_bpms = music_details[len(music_details) - 4][4:].split(u"～") # 謎のチルダで分割
            if len(music_bpms) == 1:
                music_bpms = np.concatenate((music_bpms, music_bpms))
                         
            key_names = keyinfo[np.arange(1, len(keyinfo) - 3, 3)]
            key_numbers = keyinfo[np.arange(2, len(keyinfo) - 3, 3)] #.astype(np.float64)
            total_note = np.array([keyinfo[-2].copy()])
            
            type_names = graph[np.arange(1, len(graph) - 4, 4)]
            type_points = graph[np.arange(3, len(graph) - 4, 4)] #.astype(np.float64)
            play_time = np.array([graph[-2].copy()])
            
            max_nps = np.array([both_notes_per_second.max().round(2)])
            max_bw_nps = np.array([bw_notes_per_second.max().round(2)])
            max_s_nps = np.array([s_notes_per_second.max().round(2)])
            
            mean_nps = np.array([both_notes_per_second.mean().round(2)])
            mean_bw_nps = np.array([bw_notes_per_second.mean().round(2)])
            mean_s_nps = np.array([s_notes_per_second.mean().round(2)])
            
            var_nps = np.array([both_notes_per_second.var(ddof=1).round(2)])
            var_bw_nps = np.array([bw_notes_per_second.var(ddof=1).round(2)])
            var_s_nps = np.array([s_notes_per_second.var(ddof=1).round(2)])
        
        
            I_column = np.concatenate((music_title, play_style, music_mode,
                                       music_difficulty, music_bpms, key_numbers,
                                       total_note, n_sofran, type_points,
                                       play_time,
                                       max_nps, max_bw_nps, max_s_nps,
                                       mean_nps, mean_bw_nps, mean_s_nps,
                                       var_nps, var_bw_nps, var_s_nps), axis=0)[:, None].T
            """"
            print(I)
            print(I.shape)
            print("")
            print(I_column)
            print(I_column.shape)
            """
            print(I_column.tolist())
            
            I = np.concatenate((I, I_column), axis=0)
            
        
        except TimeoutException:
            print("no alert")
    
    browser.close()
    browser.quit()

    I_df = pd.DataFrame(I)
    
    I_df.columns = ['タイトル', 'プレイスタイル', '譜面モード', '難易度'] + \
                   ['MINBPM', 'MAXBPM'] + \
                   ['S', '1', '2', '3', '4', '5', '6', '7'] + \
                   ['TOTALNOTES'] + \
                   ['ソフラン回数'] + \
                   ['乱打', '同時', '階段', 'トリ', '縦連', '皿', 'CN'] + \
                   ['PLAYTIME'] + \
                   ['MAXNPS', 'MAXNPS_BW', 'MAXNPS_S'] + \
                   ['MEANNPS', 'MEANNPS_BW', 'MEANNPS_S'] + \
                   ['VARNPS', 'VARNPS_BW', 'VARNPS_S']
                   

    I_df.values[:, 3:16] = I_df.values[:, 3:16].astype('int')
    I_df.values[:, 16:] = I_df.values[:, 16:].astype('float')

    I_df.to_csv('%s'%fname, encoding='utf-8', index=False)
    
    return I_df


def load_official_csv(csvpath='./20170325.csv'):
    
    official_data = pd.read_csv(csvpath)
    #official_data = pd.read_csv('./6282-2940_sp_score.csv')
    
    difficulty = 12
    
    nor = official_data[['タイトル', 'NORMAL 難易度', 'NORMAL クリアタイプ', 'NORMAL DJ LEVEL']].loc[official_data['NORMAL 難易度'] >= difficulty]
    nor.columns = ['タイトル', '難易度', 'クリアタイプ', 'DJ LEVEL']
    nor['譜面モード'] = 'NORMAL'
    
    hyp = official_data[['タイトル', 'HYPER 難易度', 'HYPER クリアタイプ', 'HYPER DJ LEVEL']].loc[official_data['HYPER 難易度'] >= difficulty]
    hyp.columns = ['タイトル', '難易度', 'クリアタイプ', 'DJ LEVEL']
    hyp['譜面モード'] = 'HYPER'
    
    ano = official_data[['タイトル', 'ANOTHER 難易度', 'ANOTHER クリアタイプ', 'ANOTHER DJ LEVEL']].loc[official_data['ANOTHER 難易度'] >= difficulty]
    ano.columns = ['タイトル', '難易度', 'クリアタイプ', 'DJ LEVEL']
    ano['譜面モード'] = 'ANOTHER'
    
    z = pd.concat([nor, hyp, ano], axis=0).sort_values(by='タイトル').reset_index(drop=True)
    z = z[['タイトル', '譜面モード', '難易度', 'クリアタイプ', 'DJ LEVEL']]
    return z

def load_textage_data():
    # load textage data.
    #I = scrape_textage(mode='s', difficulty=12, fname='test_music_data.csv', n_limit=999) # execute scrape.
    
    #I_sp10 = pd.read_csv('iidx_music_data_sp_10.csv')
    I_sp11 = pd.read_csv('iidx_music_data_sp_11_adv.csv')
    I_sp12 = pd.read_csv('iidx_music_data_sp_12_adv.csv')
    I = pd.concat([I_sp11, I_sp12], axis=0)    
    return I

def fix_spell(I, z):
    
    ### LongestCommonStrings 解を用いたタイトルの表記揺れ修正 ---------------------------------------------
    import sys,os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../dynamic_programming')
    import dp
    
    #X_list = 10 * ['Aegis', 'ALBIDA', 'Answer', 'Anthem Landing', 'Believe in Me', 'BLUST oF WIND', 'Bounce Bounce Bounce', 'Breaking the ground', 'Broken', 'Broken Eden', 'Dances with Snow Fairies']
    #Y_list = 10 * ['Aegis', 'ALBIDA', 'Answer', 'ANTHEM LANDING', 'Believe In Me', 'BLUST OF WIND', 'Bounce Bounce Bounce', 'Breaking the ground', 'Broken', 'BROKEN EDEN', 'Dances with Snow Fairies']
    
    X_list = z.sort_values(by=['タイトル', '譜面モード', '難易度'], ascending=True, inplace=False, kind='heapsort').reset_index(drop=True)
    Y_list = I.sort_values(by=['タイトル', '譜面モード', '難易度'], ascending=True, inplace=False, kind='heapsort').reset_index(drop=True)
    
    s = time.time()
    result3 = dp.FFMLS(Y_list['タイトル'].tolist()).complete_match_ffmls_parallel(X_list['タイトル'].tolist())
    print('time to process3: %.2f [sec]' %(time.time() - s))
    
    z = z.sort_values(by=['タイトル', '譜面モード', '難易度'], ascending=True, inplace=False).reset_index(drop=True)
    #I = I.sort_values(by=['タイトル'], ascending=True, inplace=False).reset_index(drop=True)
    
    z['タイトル'] = result3.copy()
    ### ------------------------------------------------------------------------------------------

    return z


def separate_by_cleartype(Iz):
    
    condition_train_lamp = (Iz['クリアタイプ'] == Iz['クリアタイプ']) & (Iz['クリアタイプ'] != 'NO PLAY')
    condition_test_lamp = (Iz['クリアタイプ'] != Iz['クリアタイプ']) | (Iz['クリアタイプ'] == 'NO PLAY')
    
    X_train = Iz[condition_train_lamp].iloc[:, 4:-2].values
    X_test = Iz.iloc[:, 4:-2].values.copy()#Iz[condition_test_lamp].iloc[:, 4:-2]
    y_train = Iz[condition_train_lamp].iloc[:, -2]
    y_test = Iz.iloc[:, -2].copy()#Iz[condition_test_lamp].iloc[:, -2]
    
    return X_train, y_train, X_test, y_test


def scale(X):
    from sklearn.preprocessing import StandardScaler
    standard = StandardScaler()
    X = standard.fit_transform(X) # データを標準化
    return X
    
def set_classifier():
   
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn import svm
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    # 多分最適値
    C_rbf = 10 # rbf
    g = 0.001 # rbf
    
    #C_rbf = 30 # rbf # 最高正答率 80%
    #g = 0.05 # rbf
    
    #C_rbf = 1000 # 誤差許容範囲
    #g = 0.000005
    
    C_linear = 0.02 # linear
    C_linear = 1 # linear
    
    from sklearn import mixture
    
    #clf = QuadraticDiscriminantAnalysis()
    #clf = SVC(C=C_rbf, gamma=g, probability=True, kernel='rbf', decision_function_shape='ovr', shrinking=True, class_weight='balanced', random_state=0)
    clf = SVC(C=C_linear, gamma=g, probability=True, kernel='linear', decision_function_shape='ovr', shrinking=True, class_weight='balanced', random_state=0)
    #clf = GaussianNB(priors=None)
    #clf = mixture.GMM(n_components=7, covariance_type='full') 
    
    # データの標準化とSVMを定義
    pipeline = Pipeline([
            ('std', StandardScaler(with_mean=True, with_std=True)),
            ('clf', clf)])

    clf = pipeline    
    return clf

def predict_lamp(clf, X_train, y_train, X_test, y_test):
        
    # クリアランプの学習 --------------------------------------------------------------------------------

    clf.fit(X_train, y_train.map(dic_trans))
    
    pred = pd.DataFrame(clf.predict(X_test)).iloc[:, 0].map(dic_inver)
    
    proba = clf.predict_proba(X_test)
    personal_difficulty = pd.DataFrame(100 * proba @ existent_lamp_values[::-1] / n_existent_lamp)

    """
    Iz['推定クリアタイプ'] = np.nan
    #Iz['推定クリアタイプ'][condition_test] = pred
    Iz['推定クリアタイプ'] = pred
    
    Iz['推定個人的クリア難易度'] = np.nan
    Iz['推定個人的クリア難易度'] = pd.DataFrame(100 * clf.predict_proba(X_test) @ existent_lamp_values[::-1] / n_existent_lamp)
    #Iz.sort_values(by=['推定個人的難易度'], inplace=True, ascending=False)
    
    #print(Iz[condition_test].T)
    #print(Iz)
    """
    return pred, proba, personal_difficulty

def plot_personal_difficulty(Iz):
    plt.figure()
    plt.plot(Iz['推定個人的クリア難易度'].sort_values(ascending=False, inplace=False).values)
    plt.show()

def cross_validation(clf, X_train, y_train, cv=5):
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, X_train, y_train.map(dic_trans), cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def difficulty_on_lamps(Iz):
    # ランプ別 難易度 作成
    for i, lamp in enumerate(lamp_list):
        if i in existent_lamp_values:
            index_for_lamp_value = existent_lamp_values[::-1].index(i)
            weights_for_lamps = np.concatenate((existent_lamp_values[::-1][:index_for_lamp_value+1],
                                          np.zeros(n_existent_lamp - index_for_lamp_value -1)), axis=0)
            Iz[lamp] = (100 / np.sum(weights_for_lamps) * proba @ weights_for_lamps).round(2)
        else:
            Iz[lamp] = np.zeros([X_test.shape[0]]) 
    return Iz

def print_difficulty_on_lamps(Iz):
    # ランプ別 難易度 表示
    for i in range(len(lamp_list)):
        print(Iz[(Iz['難易度']==12) & (Iz['クリアタイプ'].map(dic_trans) < dic_trans[lamp_list[i]])
                ].sort_values(by=lamp_list[i], ascending=True, inplace=False)[['タイトル', lamp_list[i]]])

def print_relative_difficulty(Iz):
    # 相対難易度 表示
    """
    for i in range(len(lamp_list)):
        print(lamp_list[i])
        print(Iz[(Iz['難易度'] == 12) & (Iz['クリアタイプ'].map(dic_trans) < dic_trans[lamp_list[i]])
                ].sort_values(by='相対難易度', ascending=True, inplace=False)[['タイトル', '相対難易度']])
    """
    print(Iz[['タイトル', 'プレイスタイル', '譜面モード', '相対難易度']][Iz['難易度'] == 12].sort_values(by='相対難易度', ascending=True, inplace=False))

    
def absolutely_difficulty(Iz, X_test):        
    Iz['絶対難易度'] = ((X_test[:, 2:] - X_test[:, 2:].mean(axis=0)) / X_test[:, 2:].std(axis=0, ddof=1)).mean(axis=1)
    return Iz

def print_absolutely_difficulty(Iz):
    print(Iz[['タイトル', 'プレイスタイル', '譜面モード', '絶対難易度']][Iz['難易度'] == 12].sort_values(by='絶対難易度', ascending=True, inplace=False))

def set_cluster(n_clusters=20):
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.mixture import BayesianGaussianMixture
    #clu = KMeans(n_clusters=n_clusters)
    #clu = SpectralClustering(n_clusters=n_clusters, eigen_solver=None,
    #                                random_state=None, n_init=10, gamma=1,
    #                                affinity='rbf', n_neighbors=10,
    #                                eigen_tol=0.0,
    #                                assign_labels='kmeans', degree=3, coef0=1,
    #                                kernel_params=None, n_jobs=-1)
    clu = BayesianGaussianMixture(n_components=n_clusters, random_state=None,
                                  max_iter=10000)
    return clu

def check_existent_lamp(Iz):
    
    existent_lamp_list = []
    existent_lamp_values = []
    for i, lamp in enumerate(lamp_list):
        if lamp in Iz['クリアタイプ'].values:
            existent_lamp_list.append(lamp)
            existent_lamp_values.append(i)
    return existent_lamp_list, existent_lamp_values

def min_max_normalization(x, axis=None):
    x = x.values
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    result = (x - x_min) / (x_max - x_min)
    return result

if __name__ == '__main__':
    
    # I を更新したいときだけ実行．
    # I = scrape_textage(mode='s', difficulty=12, fname='test_music_data.csv')
    
    I = load_textage_data()
    z = load_official_csv(csvpath='./20170325.csv')
    
    z = fix_spell(I, z) # zの曲名表記を, Iの曲名表記に合わせる.
    
    # [taxtage データ: I] と [公式 csv データ: z] の マージを行う
    Iz = pd.merge(I, z, on=['タイトル', '譜面モード', '難易度'], how='left')
        
    existent_lamp_list, existent_lamp_values = check_existent_lamp(Iz)
    n_existent_lamp = len(existent_lamp_list)
    
    X_train, y_train, X_test, y_test = separate_by_cleartype(Iz)
    X_train, X_test = scale(X_train), scale(X_test)
    
    clf = set_classifier()
    pred, proba, personal_difficulty = predict_lamp(clf, X_train, y_train, X_test, y_test)
    
    #Iz['推定クリアタイプ'] = np.nan
    Iz['推定クリアタイプ'] = pred
    
    #Iz['相対難易度'] = np.nan
    Iz['相対難易度'] = pd.DataFrame(100 * proba @ existent_lamp_values[::-1] / n_existent_lamp)
    Iz['相対難易度'] = 10 * min_max_normalization(x=Iz['相対難易度'])
    #Iz['相対難易度'] = 100 * np.exp(Iz['相対難易度'] + 1e-8) / np.exp(Iz['相対難易度'] + 1e-8).mean()
    
    Iz = difficulty_on_lamps(Iz)
    
    print_difficulty_on_lamps(Iz)
    print_relative_difficulty(Iz)
    
    
    # 絶対難易度 表示
    #Iz = absolutely_difficulty(Iz, X_test)
    #print_absolutely_difficulty(Iz)
    
    # 実際は HARD でない曲で HARD と予想される曲を表示
    #print(Iz[(Iz['クリアタイプ'].map(dic_trans) != dic_trans['HARD CLEAR']) & (Iz['推定クリアタイプ'].map(dic_trans) == dic_trans['HARD CLEAR'])][::-1][Iz['難易度']==12])
    """
    clu = set_cluster(n_clusters=20)
    cluster_id = clu.fit(Iz.iloc[:, 4:33]).predict(Iz.iloc[:, 4:33])
    
    Iz['cluster_id'] = cluster_id
    
    n_clusters = np.max(cluster_id) + 1 
    for i in range(n_clusters):
        print('cluster: %d' %i)
        print(Iz[Iz['cluster_id']==i][['タイトル', '譜面モード', '難易度', '絶対難易度']].sort_values(by=['絶対難易度']))
    """  
    
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn import manifold
    
    #dec = PCA(n_components=2, whiten=True)
    dec = TSNE(n_components=2, random_state=0)
    #dec = manifold.MDS(n_components=2, max_iter=100, n_init=7)
    #dec = manifold.Isomap(n_neighbors=3, n_components=2)
    Id = dec.fit_transform((Iz.T)['MINBPM':'PLAYTIME'].T)
    
    
    #import matplotlib.cm as cm
    cm = generate_cmap(['grey', 'purple', 'limegreen', 'mediumblue' , 'red', 'yellow', 'skyblue'])
    plt.figure()
    plt.title('actually clear lamp')
    plt.scatter(Id[:, 0], Id[:, 1], c=Iz['クリアタイプ'].map(dic_trans), cmap=cm)
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.title('predicted clear lamp')
    plt.scatter(Id[:, 0], Id[:, 1], c=Iz['推定クリアタイプ'].map(dic_trans), cmap=cm)
    plt.colorbar()
    plt.show()
    """
    

