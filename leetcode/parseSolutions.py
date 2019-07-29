# import re
import regex as re
import json
# import pdb

def getProblems(text):
    pattern = '(^### .+?)(?:^### |^## |^# |\Z)'
    flags = re.DOTALL | re.MULTILINE
    problems = re.findall(pattern, text, flags, overlapped=True)
    return problems

def parseProblem(problem):
    question = {}
    flags = re.DOTALL | re.MULTILINE
    m = re.search('^### ([0-9]+). ([^\n]+)\n', problem, flags)
    question['qid'] = int(m.group(1).strip())
    question['title'] = m.group(2).strip()
    m = re.search('- \[Link\]\((https://leetcode\.com/[a-z0-9\./-]+)\)', problem, flags)
    question['link'] = m.group(1).strip()
    m = re.search('- Tags:([^\n]+)\n', problem, flags)
    question['tags'] = list(map(lambda tag: tag.strip(), m.group(1).strip().split(',')))
    m = re.search('- Stars: ([1-7])(.+?)(?:^#|\Z)', problem, flags, overlapped=True)
    question['difficulty'] = int(m.group(1))
    question['comment'] = m.group(2).strip()
    question['reviews'] = ["2019-02-01T15:33:17.821Z"]
    return question

def getSolutions(problem):
    pattern = '(^#### .+?)(?:^#### |^### |^## |^# |\Z)'
    flags = re.DOTALL | re.MULTILINE
    solutions = re.findall(pattern, problem, flags, overlapped=True)
    return solutions

def parseSolution(solution):
    parsedSol = {}
    flags = re.DOTALL | re.MULTILINE
    m = re.search('^#### ([^\n]+)\n(.+)\Z', solution, flags)
    parsedSol['title'] = m.group(1).strip()
    parsedSol['text'] = m.group(2).strip()
    return parsedSol

if __name__ == '__main__':

    dct = {}
    
    with open('./notes.md') as hd:
        text = hd.read()

    dct['text'] = text

    problems = getProblems(text)
    dct['questions'] = []
    for problem in problems:
        question = parseProblem(problem)

        solutions = getSolutions(problem)
        question['solutions'] = list(map(parseSolution, solutions))
        dct['questions'].append(question)

    jsonStr = json.dumps(dct)
    with open('./static.json', 'w') as hd:
        hd.write(jsonStr)
        
