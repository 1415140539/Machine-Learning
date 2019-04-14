import re
import sys
import time
import random
import logging
import requests
import mypymysql as my
from bs4 import BeautifulSoup

logger = logging.getLogger("spider")
fomatter= logging.Formatter("%(asctime)s  %(message)s %(levelname)s")

file_handler = logging.FileHandler("spider.log")
file_handler.setFormatter(fomatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(console_handler)

logger.setLevel(logging.INFO)


logger.addHandler(file_handler)
logger.addHandler(console_handler)


NORMAL_CODE = 200
ALL_PAGE = 3
CLIENT_ERROR_MIN = 400
CLIENT_ERROR_MAX = 500
SERVER_ERROR_MAX = 600
UA_pool = [{"User-Agent":'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0'},
    {'User-Agent':"Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; en) Opera 9.50"},
    {'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.57.2 (KHTML, like Gecko) Version/5.1.7 Safari/534.57.2"},
    {"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36"},
    {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11"},
    {"User-Agent":"Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.133 Safari/534.16"},
    {"User-Agent":"Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E"},
    {"User-Agent":"Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SV1; QQDownload 732; .NET4.0C; .NET4.0E; SE 2.X MetaSr 1.0)"}
    ]
def get_one_page_html(url,times):
    if times == 0:
        return
    headers = random.choice(UA_pool)
    res = requests.get(url,headers = headers)
    if res.status_code == NORMAL_CODE:
        return res.text
    elif CLIENT_ERROR_MIN<res.status_code <CLIENT_ERROR_MAX:
        #客户端异常
        logger.error("Client error")
    elif CLIENT_ERROR_MAX < res.status_code < SERVER_ERROR_MAX:
        logger.info("Server error")
        #多次尝试
        time.sleep(random.randint(1,3))
        get_one_page_html(url,times-1)
def get_message(html):
    l = []
    job = re.findall('<div class="job-title">([\s\S]+?)<[\s\S]+<span class="red">([\s\S]+?)<',html)[0]
    soup = BeautifulSoup(html,"html.parser")
    region = re.findall('<p>([\s\S]+?)<em class="vline"></em>([\s\S]+?)<em class="vline"></em>([\s\S]+?)</p>',html)[0]
    conmany = soup.find_all(attrs={"class":"company-text"})[0]
    conmany = conmany.find("p").get_text()
    img = re.findall('<h3 class="name">[\s\S]+?>([\s\S]+?)<',html)[1:]
    elem =img[0] +"|"+ job[0] + "|" +job[1] + "|" + conmany + "|" +"|".join(region)
    l.append(elem)
    return l

def get_detail(html):
    soup = BeautifulSoup(html,"html.parser")
    divs =soup.find_all(attrs = {"class":"job-primary"})
    l = []
    for div in divs:
        l.append(get_message(str(div)))
    return l
def write_to_sql(l,table):
    My_py = my.my_py(db = "job")
    connect = My_py.connect()
    for i in l:
        list = i[0].split("|")
        comany,job,salary,team,region,request,recore = tuple(list)
        sql = "insert into "+table+" (comany, job, salary, team," \
              " region, request, recore) values (%s,%s,%s,%s,%s,%s,%s)"
        params = (comany,job,salary,team,region,request,recore)
        My_py.execute(sql,params)
    My_py.myclose()
def main():
    for i in range(1,ALL_PAGE+1):
        url = "https://www.zhipin.com/c101010100/?query=python%E5%AE%9E%E4%B9%A0&page={}&ka=page-{}".format(i,i)
        html = get_one_page_html(url,5)
        l = get_detail(html)
        print(l)
        write_to_sql(l,'found_job')
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)
if __name__ == "__main__":
    main()