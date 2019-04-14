import pymysql
import logging


logger = logging.getLogger("mylogger")

Fomatter = logging.Formatter('%(asctime)s %(message)s %(levelname)s')

File_Handler = logging.FileHandler("mylogger.log")
File_Handler.setFormatter(Fomatter)

logger.setLevel(logging.DEBUG)
logger.addHandler(File_Handler)
class my_py():
    def __init__(self,
                 user = "root",
                 host = "127.0.0.1",
                 password = '981227',
                 db = None,
                 port = "3306",
                 charset = "utf8"):
        self. host = host
        self.user = user
        self.password = password
        self.port = port
        self.db = db
        self.charset = charset
        self.cur = None
        self.con = None
    def connect(self):
         try:
            self.con = pymysql.connect(host = self.host,user= self.user,
                                       password = self.password,
                                       db = self.db,charset= self.charset)
            self.cur = self.con.cursor()
         except:
            logger.error("CONNECT SQL FAILED")
            logger.error("CREATE CURSOR FAILED")
            return False
         return True
    def create_db(self):
        sql = "create database if not exists " + self.db + " charset utf8 collate utf8_general_ci"
        try:
            print(type(self.cur))
            self.cur.execute(sql)
            print(self.cur)
        except:
            logger.error("CREATE DB FAILED")
            return False
        return True
    def execute(self,sql,params = None):
        if self.connect():
            try:
                if self.con and self.cur:
                    self.cur.execute(sql,params)
                    self.con.commit()
            except Exception as err:
                print(err)
                logger.error('EXECUTE FAILED')
                return False
            return  True
        else:
            logger.error("FETCH FAILED")
            return False
    def fetchCount(self,sql,params = None):
        if self.connect() == False:
            return False
        else:
            try:
                if self.cur and self.con:
                    self.execute(sql,params)
            except:
                return False
            return self.cur.fetchone() #操作返回数据库得到的一条数据
        #返回的是一个元组 ， 所以取第一个值
    def myclose(self):
        if self.cur:
            self.cur.close()
        if self.con:
            self.con.close()
        return True

# if __name__ == "__main__":
#     ql = my_py(db = "bink")
#
#     # sql = "insert into user(name,passwd) values(%s,%s)"
#     # params = ("yuange",'123456')
#     # print(ql.execute(sql,params))
#     # sql = "select count(*) from user"
#     # print(ql.fetchCount(sql))
#     # print(ql.myclose())
#     logger.removeHandler(File_Handler)