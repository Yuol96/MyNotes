#%%
import requests

#%%
def getQuestionList():
    url = 'https://leetcode.com/api/problems/algorithms/'
    res = requests.get(url).json()
    qList = res['stat_status_pairs']
    return qList

#%%
def generateLink(slug):
    return 'https://leetcode.com/problems/{}/'.format(slug)

def convertDifficulty(num):
    if num == 1:
        return 'Easy'
    if num == 2:
        return 'Medium'
    if num == 3:
        return 'Hard'
    raise ValueError('Invalid Difficulty Input: {}'.format(num))

def outputQList(qList, fpath, filtFunc=None, sortFunc=None):
    if filtFunc is not None:
        qList = list(filter(filtFunc, qList))
    if filtFunc is not None:
        qList.sort(key=sortFunc)
    with open(fpath, 'w') as hd:
        for q in qList:
            hd.write('- [ ] [{}. {}]({}), {}%, {}\n\n'.format(q['stat']['question_id'], q['stat']['question__title'], generateLink(q['stat']['question__title_slug']), round(q['stat']['total_acs']/q['stat']['total_submitted']*100, 2), convertDifficulty(q['difficulty']['level'])))

#%%
qList = getQuestionList()
fpath = './leetcode/parse/300-399.md'
filtFunc = lambda q: q['stat']['question_id'] >= 300 and q['stat']['question_id'] < 400
sortFunc = lambda q: (q['difficulty']['level'], -q['stat']['total_acs']/q['stat']['total_submitted'])
outputQList(qList, fpath, filtFunc, sortFunc)

#%%
