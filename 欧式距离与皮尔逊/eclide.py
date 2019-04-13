import os
import sys
import numpy as np
import json


def read_data(filename):
    with open(filename, "r") as f:
        ratings = f.read()
    return json.loads(ratings)
def cacl_es(ratings, user1, user2):
    movies = set()
    for movie in  ratings[user1]:
        if movie in ratings[user2]:
            movies.add(movie)
    if len(movie) == 0:
        return 0
    diffs = []
    for movie in movies:
        diffs.append(np.square(ratings[user1][movie] - ratings[user2][movie]))
    diffs = np.array(diffs)
    euclidean_score = 1 / (1 + np.sqrt(diffs.sum()))
    return euclidean_score
def eval_es(ratings):
    users, estimates = list(ratings.keys()) , []
    for user1 in users:
        esrow = []
        for user2 in users:
            esrow.append(cacl_es(ratings,user1,user2))
        estimates.append(esrow)
    users = np.array(users)
    esmat = np.array(estimates)
    return users,esmat
def main(argc, argv, envir):
    ratings = read_data("ratings.json")
    users, esmat = eval_es(ratings)
    print(users)
    print(esmat)
if __name__ == "__main__":
    sys.exit(main(len(sys.argv),sys.argv,os.environ))