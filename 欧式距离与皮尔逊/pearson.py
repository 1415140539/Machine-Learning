import os
import sys
import numpy as np
import json


def read_data(filename):
    with open(filename, "r") as f:
        return json.loads(f.read())
def calc_ps(ratings, user1,user2):
    movies = set()
    for movie in ratings[user1]:
        if movie  in ratings[user2]:
            movies.add(movie)
    n = len(movies)
    if n == 0:
        return 0
    x = np.array([ratings[user1][movie] for movie in movies])
    y = np.array([ratings[user2][movie] for movie in movies])
    sx = x.sum()
    sy = y.sum()
    xx = (x**2).sum()
    yy = (y**2).sum()
    xy = (x*y).sum()
    sxx = xx -sx**2/n
    syy = yy - sy**2/n
    sxy = xy - sx*sy/n
    if sxy *syy == 0:
        return 0
    pearson_score = sxy / np.sqrt(sxx * syy)
    return pearson_score
def eval_ps(ratings):
    users, estimate = list(ratings.keys()), []
    for user1 in ratings:
        esmate = []
        for user2 in ratings:
            esmate.append(calc_ps(ratings, user1, user2))
        estimate.append(esmate)
    return users, estimate
def main(argc, argv, envir):
    ratings = read_data("ratings.json")
    users, estimaes = eval_ps(ratings)
    print(users)
    for es in estimaes:
        print(es)
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv, os.environ))