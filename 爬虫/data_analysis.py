import os
import re
import sys
import platform
import numpy as np
import mypymysql as my
import sklearn.metrics as ms
import matplotlib.pyplot as mp
import sklearn.linear_model as sl
import sklearn.preprocessing as sp
import sklearn.model_selection as sm
mp.rcParams['font.sans-serif']=['SimHei']
mp.rcParams['axes.unicode_minus']=False


def read_data(db,table):
    My_py = my.my_py(db = db)
    My_py.connect()
    sql = "select * from %s"%table
    My_py.execute(sql)
    notes = My_py.cur.fetchall()
    companies, jobs , salarys, teams, regions, requests, recores = \
    [],[],[],[],[],[],[]
    for note in notes:
        company, job, salary, team, region, request, recore = tuple(note[1:])
        companies.append(company)
        jobs.append(job)
        salarys.append(sum([int(substr) for substr in re.findall("(\\d)+k-(\\d+)k",salary)[0]])/2)
        teams.append(int(re.findall("(\\d+)人",team)[0]))
        regions.append(region)
        requests.append(request)
        recores.append(recore)
    return (np.array(companies),np.array(jobs),np.array(salarys),
            np.array(teams),np.array(regions),np.array(requests),
            np.array(recores))
def analy_companies(companies):
    company_list = np.unique(companies)
    L  = []
    for company in company_list:
        L.append((company == companies).sum())
    sorted_indices = np.array(L).argsort()[-4:]
    company = np.array(company_list)[sorted_indices]
    init_chart()
    mp.pie(np.array(L)[sorted_indices],[0.1,0.3,0.2,0.1],
           company,["blue","red","yellow","limegreen"],
           shadow=True, startangle=90)
def analy_regin(regions):
    region_list = np.unique(regions)
    L  = []
    for region in region_list:
        L.append((region == regions).sum())
    intervel = [0.1] * len(L)
    mp.subplot(223)
    mp.title("Region Pie", fontsize = 26)
    mp.pie(np.array(L),intervel,
           region_list,
           shadow=True, startangle=90)
def analy_recores(recores):
    recore_list = np.unique(recores)
    L  = []
    for recore in recore_list:
        L.append((recore == recores).sum())
    mp.subplot(222)
    mp.title("Recore Pie", fontsize = "26")
    mp.pie(L,[0.1,0.2,0.1,0.1],recore_list,["red","blue","green","yellow"],
           "%d%%",shadow=True,startangle=90)
def analy_job(jobs):
    job_list = np.unique(jobs)
    L  = []
    for job in job_list:
        L.append((job == jobs).sum())
    sorted_index = np.array(L).argsort()[-10:]
    mp.subplot(224)
    mp.title("Job Pie", fontsize = "26")
    intervel = [0.1] * sorted_index.size
    mp.pie(np.array(L)[sorted_index],intervel,np.array(job_list)[sorted_index],shadow=True,startangle=90)
def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 /255)
    mp.subplot(221)
    mp.title("Company Pie", fontsize = "26")
def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()
def make_data():
    list = np.array([
        ["经验不限","20",'本科',"0"],
        ["1年以内","5.6","本科","1"],
        ["应届生","4.5","本科","0"],
        ["经验不限","4.5","本科","1"],
        ["1年以内","6.6","硕士","1"],
        ["经验不限","10","硕士","1"],
        ["1-3年","7.5","本科","1"],
        ["经验不限","6.6","本科","1"],
        ["应届生","5.6","本科","0"],
        ["应届生","2.5","本科","1"],
        ["1年以内","6.6","本科","1"],
        ["经验不限","12","硕士","0"],
        ["经验不限","20","博士","1"],
        ["经验不限","23", "博士", "0"]
    ])
    return list
def deal_data(data):
    salays = []
    result = []
    cols = data.shape[1]
    data = data.T
    L = []
    encoder_ex = []
    for col in range(cols):

        if col == 1:
            salays.append([float(substr) for substr in data[col]])
        if col == 3:
            result.append([int(substr) for substr in data[col]])
        if col == 0 or col == 2:
            encoder = sp.LabelEncoder()
            ex = encoder.fit_transform(data[col])
            L.append(ex)
            encoder_ex.append(encoder)
    data = np.array((L[0],salays[0],L[1],result[0])).T
    y = data[:,-1]
    x = data[:,:-1]
    return x,y,encoder_ex
def deal_test(data,encoders):
    salays = []
    cols = data.shape[1]
    data = data.T
    L = []
    for col in range(cols):
        if col == 1:
            salays.append([float(substr) for substr in data[col]])
        if col == 0:
            ex = encoders[0].fit_transform(data[col])
            L.append(ex)
        elif col ==2:
            ex = encoders[1].fit_transform(data[col])
            L.append(ex)
    data = np.array((L[0],salays[0],L[1])).T
    return data
def train_model(x,y):
    model = sl.LogisticRegression(max_iter=100,C = 10)
    model.fit(x,y)
    return model
def make_test_data():
    data = np.array([
        ["经验不限","4.5","本科"],
        ["一年以内","4.6","本科"],
        ["一年以内", "9.2", "硕士"],
        ["经验不限", "20", "本科"]
    ])
    return data
def pred_model(model,test_x):
    return model.predict(test_x)
def main():
    companies, jobs, salarys, teams, regions, requests, recores = read_data("job","found_job")
    analy_companies(companies)
    analy_recores(recores)
    analy_regin(regions)
    analy_job(jobs)
    data = make_data()
    x, y,encoder_ex = deal_data(data)
    model = train_model(x,y)
    test_x = make_test_data()
    test_x = deal_test(test_x,encoder_ex)
    pred_y = pred_model(model,test_x)
    show_chart()
    return 0

if __name__ == "__main__":
    main()