#%%
import requests
from pathlib import Path
import re

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

def outputQList(qList, fpath, resumeSet, filtFunc=None, sortFunc=None):
    if filtFunc is not None:
        qList = list(filter(filtFunc, qList))
    if filtFunc is not None:
        qList.sort(key=sortFunc)
    print('Output {} questions in {}'.format(len(qList), fpath))
    with open(fpath, 'w') as hd:
        for q in qList:
            hd.write('- [{}] [{}. {}]({}), {}{}%, {}\n\n'.format('x' if q['stat']['question_id'] in resumeSet else ' ', q['stat']['question_id'], q['stat']['question__title'], generateLink(q['stat']['question__title_slug']), '**Paid Only**, ' if q['paid_only'] else '', round(q['stat']['total_acs']/q['stat']['total_submitted']*100, 2), convertDifficulty(q['difficulty']['level'])))

#%%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=100000)
    parser.add_argument('--resume')
    args = parser.parse_args()

    resumeFile = Path('./leetcode/parse/{}'.format(args.resume))
    resumeSet = set()
    if args.resume and resumeFile.exists():
        with open(resumeFile) as hd:
            for line in hd:
                if not line:
                    continue
                m = re.search('- \[(.)\] \[([0-9]+)\.', line)
                if m and m.group(1) == 'x':
                    qid = int(m.group(2))
                    resumeSet.add(qid)
        print('Resumed from {} in which you have solved {} questions'.format(resumeFile, len(resumeSet)))

    qList = getQuestionList()
    fpath = './leetcode/parse/{:03d}-{:03d}.md'.format(args.start, args.end)
    filtFunc = lambda q: q['stat']['question_id'] >= args.start and q['stat']['question_id'] <= args.end
    sortFunc = lambda q: (q['difficulty']['level'], -q['stat']['total_acs']/q['stat']['total_submitted'])
    outputQList(qList, fpath, resumeSet, filtFunc, sortFunc)

#%%
