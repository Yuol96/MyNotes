from IPython import embed
from pprint import pprint

import time
from urllib.request import urlretrieve
import os
import requests

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from browsermobproxy import Server

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select

import re
from tqdm import tqdm
from pathlib import Path
import pickle
import pdb

cfg = {
    'base_url': 'https://leetcode.com/problemset/algorithms/',
}

def input_keys(elem, inp, clear=True):
    if clear:
        elem.clear()
    elem.send_keys(inp)

def login():
    try:
        driver.get(cfg['base_url'])

        time.sleep(3)

        difficulty = driver.find_element_by_xpath('//*[@id="question-app"]/div/div[2]/div[2]/div[2]/table/thead/tr/th[6]')
        difficulty.click()
        difficulty.click()

        rowsPerPage = driver.find_element_by_xpath('//*[@id="question-app"]/div/div[2]/div[2]/div[2]/table/tbody[2]/tr/td/span[1]/select')
        # driver.execute_script("arguments[0].scrollIntoView(false);", element)  # 移动到元素element对象的“底端”与当前窗口的“底部”对齐 
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")  # 移动到页面最底部 
        rowsPerPage.click()
        rowsPerPage.find_element_by_xpath('option[4]').click()
        # action = ActionChains(driver).move_to_element(rowsPerPage)
        # action.click()
        # action.send_keys(Keys.ARROW_DOWN)
        # action.send_keys(Keys.ARROW_DOWN)
        # action.send_keys(Keys.ENTER)
        # action.perform()

        # driver.find_element_by_xpath('/html/body/div[1]/div[22]/ul[2]/li[3]/a').click()
        # driver.find_element_by_xpath('/html/body/div[1]/div[22]/ul[2]/li[3]/ul/li[3]').click()

        # selectdb = driver.find_element_by_xpath('//*[@id="select.database.stripe"]/div/div[1]/span[2]')
        # action = ActionChains(driver).move_to_element(selectdb)
        # action.click()
        # action.send_keys(Keys.ARROW_DOWN)
        # action.send_keys(Keys.ENTER)
        # action.perform()

        # beiyin = driver.find_element_by_xpath('/html/body/div[9]/div/ul/li[2]/a')
        # beiyin.click()

        # driver.find_element_by_xpath('//*[@id="WOS_CitedReferenceSearch_input_form"]/div[2]/div[2]/span').click()

        # checkboxes = driver.find_element_by_xpath('//*[@id="WOS_CitedReferenceSearch_input_form"]/div[2]/div[2]/div/div/div[1]').find_elements_by_tag_name('input')
        # # assert len(checkboxes) == 8
        # for i in range(8):
        #     box = checkboxes[i]
        #     if box.is_selected() and i>0:
        #         box.click()

    except Exception as e:
        print(e)
        print('---- Login Again! ----')
        login()

# def search(title):
#     box = driver.find_element_by_name('value(input2)')
#     input_keys(box, title)

#     selectitem = driver.find_element_by_xpath('//*[@id="searchrow2"]/td[3]/span/span[1]/span')
#     action = ActionChains(driver).move_to_element(selectitem)
#     action.click()
#     action.send_keys('Cited Title')
#     action.send_keys(Keys.ENTER)
#     action.perform()

#     driver.find_element_by_xpath('//*[@id="searchCell3"]/span[1]/button').click()

# def collect_article_info(title):
#     dct = {'exists': True}
#     try:
#         tbody = driver.find_element_by_xpath('//*[@id="records_chunk"]')
#     except:
#         print('No search results')
#         dct['exists'] = False
#         return dct
#     rows = tbody.find_elements_by_tag_name('tr')
#     if len(rows) > 1:
#         idx = int(input('Please indicate which row should be used'))
#         if idx == -1:
#             print('No clickable links')
#             dct['exists'] = False
#             return dct
#         row = rows[idx]
#     elif len(rows) == 0:
#         print('No search results')
#         dct['exists'] = False
#         return dct
#     else:
#         row = rows[0]

#     row.find_element_by_xpath('//*[@id="show_author_exp_link_1"]/a').click()
#     dct['authors'] = row.find_element_by_xpath('//*[@id="author_exp_1"]').text.strip()
#     dct['source'] = row.find_element_by_xpath('//*[@id="cited_work_abbrev_1"]').text.strip()
#     dct['year'] = row.find_element_by_xpath('//*[@id="records_chunk"]/tr/td[5]').text.strip()
#     dct['volume'] = row.find_element_by_xpath('//*[@id="records_chunk"]/tr/td[6]').text.strip()
#     dct['no'] = row.find_element_by_xpath('//*[@id="records_chunk"]/tr/td[7]').text.strip()
#     dct['page'] = row.find_element_by_xpath('//*[@id="records_chunk"]/tr/td[8]').text.strip()
#     dct['doi'] = row.find_element_by_xpath('//*[@id="exp_identifier_1"]').text.replace('DOI:','').strip()
#     dct['num_cited'] = row.find_element_by_xpath('//*[@id="records_chunk"]/tr/td[10]/a').text.strip()
#     dct['title'] = title

#     return dct

# def collect_info(idx):
#     driver.find_element_by_xpath('/html/body/div[1]/div[26]/div/div/div/div[3]/div[1]/span/button/i').click()

#     main_hd = driver.current_window_handle

#     # 弹窗
#     dropdown = driver.find_element_by_xpath('//*[@id="ui-id-7"]/form/div[2]/div[2]/span/span[1]/span')
#     action = ActionChains(driver).move_to_element(dropdown)
#     action.click()
#     action.send_keys(Keys.ARROW_UP)
#     action.send_keys(Keys.ARROW_UP)
#     action.send_keys(Keys.ARROW_UP)
#     action.send_keys(Keys.ARROW_UP)
#     action.send_keys(Keys.ARROW_DOWN)
#     action.send_keys(Keys.ARROW_DOWN)
#     action.send_keys(Keys.ENTER)
#     action.perform()
#     driver.find_element_by_xpath('//*[@id="ui-id-7"]/form/div[3]/span/button').click()

#     print_hd = driver.window_handles[1]
#     records = get_print_records(main_hd, print_hd)
#     try:
#         assert len(records) == 1
#     except Exception as e:
#         embed()
#         raise e

#     record = records[0]

#     # info = ['{}: {}'.format(k, record[k]) for k in ['Title', 'Author(s)', *record['keywords'], 'ISSN', 'Times Cited in Web of Science Core Collection', 'Total Times Cited']]
#     # info = ['Record {} of 60'.format(idx)] + info
#     # info = '\n'.join(info)

#     return record

# def collect_cited_info():
#     a = driver.find_element_by_xpath('//*[@id="sidebar-column1"]/div[1]/div[3]/div/a')
#     driver.execute_script('window.open("{}")'.format(a.get_property('href')))

#     main_hd = driver.current_window_handle
#     citation_hd = driver.window_handles[1]
#     driver.switch_to.window(citation_hd)

#     # dropdown = driver.find_element_by_xpath('//*[@id="numberOfRec_per_page_bottom"]/span[2]/span[1]/span')
#     # action = ActionChains(driver).move_to_element(dropdown)
#     # action.click()
#     # action.send_keys(Keys.ARROW_DOWN)
#     # action.send_keys(Keys.ARROW_DOWN)
#     # action.send_keys(Keys.ARROW_DOWN)
#     # action.send_keys(Keys.ENTER)
#     # action.perform()

#     records = []
#     WebDriverWait(driver,20,0.5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="pageCount.top"]')))

#     num_pages = int(driver.find_element_by_xpath('//*[@id="pageCount.top"]').text)
#     for __ in range(num_pages):
#         driver.find_element_by_xpath('//*[@id="SelectPageChkId"]').click()
#         driver.find_element_by_xpath('//*[@id="page"]/div[1]/div[26]/div[2]/div/div/div/div[2]/div[3]/div[3]/div[2]/span[1]/button/i').click()

#         # 弹窗
#         dropdown = driver.find_element_by_xpath('//*[@id="ui-id-9"]/form/div[2]/div[2]/span/span[1]/span')
#         action = ActionChains(driver).move_to_element(dropdown)
#         action.click()
#         action.send_keys(Keys.ARROW_UP)
#         action.send_keys(Keys.ARROW_UP)
#         action.send_keys(Keys.ARROW_UP)
#         action.send_keys(Keys.ARROW_UP)
#         action.send_keys(Keys.ARROW_DOWN)
#         action.send_keys(Keys.ARROW_DOWN)
#         action.send_keys(Keys.ENTER)
#         action.perform()
#         driver.find_element_by_xpath('//*[@id="ui-id-9"]/form/div[3]/span/button').click()

#         print_hd = driver.window_handles[2]

#         records.extend(get_print_records(citation_hd, print_hd))

#         driver.find_element_by_xpath('//*[@id="summary_navigation"]/nav/table/tbody/tr/td[3]/a').click()

#     return records

# def get_print_records(main_hd, print_hd):
#     driver.switch_to.window(print_hd)

#     tables = driver.find_elements_by_tag_name('table')
#     records = []
#     for table in tables:
#         try:
#             record = {}
#             trs = table.find_elements_by_tag_name('tr')
#             if len(trs) < 10:
#                 continue
#             for i in range(len(trs)):
#                 s = trs[i].text
#                 if s.startswith('Title'):
#                     record['Title'] = s.replace('Title:', '').strip()
#                 elif s.startswith('Author(s)'):
#                     record['Author(s)'] = '; '.join(re.findall('\(([^;\(\)]+)\)(?:;|\Z)', s))
#                 elif 'Source:' in s:
#                     keywords = []
#                     for k, v in re.findall('([a-zA-Z]+):\s((?:[^\s:]+\s)+)', s):
#                         record[k] = v.strip()
#                         keywords.append(k)
#                     record['keywords'] = keywords
#                 elif s.startswith('ISSN'):
#                     record['ISSN'] = s.split(':')[1]
#                 elif s.startswith('DOI'):
#                     record['DOI'] = s.split(':')[1].strip()
#                 elif s.startswith('Times Cited in Web of Science Core Collection:'):
#                     record['Times Cited in Web of Science Core Collection'] = s.split(':')[1]
#                 elif s.startswith('Total Times Cited'):
#                     record['Total Times Cited'] = s.split(':')[1]
#             if record and 'Title' in record:
#                 records.append(record)
#         except Exception as e:
#             print(e)
#             embed()
#             continue
#     driver.close()
#     driver.switch_to.window(main_hd)
#     return records

# def convert2info(record, idx, tot):
#     info = ['{}: {}'.format(k, record[k]) for k in ['Title', 'Author(s)', *record['keywords'], 'ISSN', 'Times Cited in Web of Science Core Collection', 'Total Times Cited'] if k in record]
#     info = ['Record {} of {}'.format(idx, tot)] + info
#     info = '\n'.join(info)
#     return info

# def convert2oneline(record, idx):
#     oneline = '{}. {}. {}. {}'.format(
#         idx, 
#         record['Author(s)'].split(';')[0], 
#         ', '.join([record[k] for k in ['Source', 'Published', 'Volume', 'Issue', 'Pages'] if k in record]),
#         record.get('DOI', '')
#     )
#     return oneline.strip()

# def close_all_windows():
#     for hd in driver.window_handles[1:]:
#         driver.switch_to.window(hd)
#         driver.close()
#     driver.switch_to.window(driver.window_handles[0])

# def parse_article(idx, title):
#     close_all_windows()
#     login()
#     search(title)
#     dct = collect_article_info(title)
#     try:
#         driver.find_element_by_xpath('//*[@id="records_chunk"]/tr/td[4]/a[2]').click()
#     except:
#         print('Could not find clickable link!')
#         dct['exists'] = False
#     if not dct['exists']:
#         aritcle_record = {}
#         aritcle_info = ['Record {} of {}'.format(idx, cfg['total_num']), 'Title: ' + title, '未收录']
#         aritcle_info = '\n'.join(aritcle_info)
#         cited_records = []
#         cited_infos = []
#         part1 = aritcle_info
#         part2 = ""
#     else:
#         aritcle_record = collect_info(idx)
#         aritcle_info = convert2info(aritcle_record, idx, cfg['total_num'])

#         cited_records = collect_cited_info()
#         cited_infos = []
#         for i, record in enumerate(cited_records):
#             cited_infos.append(convert2info(record, i, len(cited_records)))
#         part1 = aritcle_info
#         part2 = '\n\n'.join([convert2oneline(aritcle_record, idx)] + cited_infos)

#     cache = {
#         'idx': idx,
#         'title': title,
#         'aritcle_record': aritcle_record,
#         'aritcle_info': aritcle_info,
#         'cited_records': cited_records,
#         'cited_infos': cited_infos,
#         'part1': part1,
#         'part2': part2,
#     }
#     return part1, part2, cache

# def generateform(caches):
#     form = []
#     for idx in range(46, 61):
#         cache = caches[idx]
#         if len(cache['aritcle_record']) == 0:
#             continue
#         record = cache['aritcle_record']
#         st = set(list(map(str.strip, record['Author(s)'].split(';'))))
#         count = 0
#         for cite in cache['cited_records']:
#             cite_set = set(list(map(str.strip, cite['Author(s)'].split(';'))))
#             if len(st & cite_set) == 0:
#                 count += 1

#         line = [
#             '{}'.format(cache['idx']),
#             record['Author(s)'].split(';')[0], 
#             ', '.join([record[k] for k in ['Source', 'Published', 'Volume', 'Issue', 'Pages'] if k in record]),
#             record.get('DOI', ''),
#             str(len(cache['cited_records'])),
#             '{}'.format(count),
#             '',
#         ]
#         line = '\t'.join(line)
#         form.append(line)

#     form = '\n'.join(form)

#     return form

def parseQuestions():
    tbody = driver.find_element_by_xpath('//*[@id="question-app"]/div/div[2]/div[2]/div[2]/table/tbody[1]')
    rows = tbody.find_elements_by_tag_name('tr')
    questions = []
    for row in tqdm(rows):
        index = int(row.find_element_by_xpath('td[2]').text.strip())
        titleA = row.find_element_by_xpath('td[3]/div/a')
        title = titleA.text.strip()
        link = titleA.get_attribute('href')
        acceptance = row.find_element_by_xpath('td[5]').text.strip()
        difficulty = row.find_element_by_xpath('td[6]/span').text.strip()
        question = {
            'index': index,
            'title': title,
            'link': link,
            'acceptance': acceptance,
            'difficulty': difficulty,
        }
        questions.append(question)
    return questions

def getFirst300(questions):
    first300 = list(filter(lambda question: question['index']<=300, questions))
    with open('./first300.md', 'w') as hd:
        for row in first300:
            hd.write('- [ ] [{}. {}]({}), {}, {}\n\n'.format(row['index'], row['title'], row['link'], row['acceptance'], row['difficulty']))
    return first300

if __name__ == '__main__':

    # cache_file = Path('./caches.pkl')
    # if cache_file.exists():
    #     caches = pickle.load(open(str(cache_file), 'rb'))
    # else:
    #     caches = {}

    cache_file = Path('./questions_difficultyOrder.pkl')
    if cache_file.exists():
        questions = pickle.load(open(str(cache_file), 'rb'))
    else:
        driver = webdriver.Chrome()
        login()
        questions = parseQuestions()
        pickle.dump(questions, open('./questions_difficultyOrder.pkl', 'wb'))

    first300 = getFirst300(questions)


    # idx = 45
    # try:
    #     with open('./titles.txt') as hd:
    #         for title in tqdm(hd):
    #             title = title.strip()
    #             if not title:
    #                 continue
    #             idx += 1
    #             if idx in caches:
    #                 continue

    #             part1, part2, cache = parse_article(idx, title)

    #             caches[idx] = cache

    #             with open('./part1.txt', 'a') as ahd:
    #                 ahd.write('\n\n\n')
    #                 ahd.write(part1)
    #             with open('./part2.txt', 'a') as ahd:
    #                 ahd.write('\n\n\n')
    #                 ahd.write(part2)


    # except Exception as e:
    #     print(e)
    #     pickle.dump(caches, open(str(cache_file), 'wb'))
    #     embed()
    #     driver.quit()
    #     import sys
    #     sys.exit('You need to DEBUG!')

    # pickle.dump(caches, open(str(cache_file), 'wb'))

    # form = generateform(caches)

    # with open('./form.txt', 'w') as hd:
    #     hd.write(form)

    print("Finished All!")

    embed()

    driver.quit()
