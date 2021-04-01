# Module Imports
import pymysql
import sys
import configparser as cf
import crypto as cp
from decimal import Decimal
# Connect to MariaDB Platform
try:
    config = cf.ConfigParser()
    config.read('./properties/db.properties')
    try:
        conn = pymysql.connect(
            user=cp.decrypt(config.get("db", "user")),
            passwd=cp.decrypt(config.get("db", "password")),
            host=config.get("db", "prd_host"),  # prd-server
            port=int(config.get("db", "port")),
            db=config.get("db", "database"),
            charset=config.get("db", "charset")
        )
    except pymysql.Error as e:
        conn = pymysql.connect(
            user=cp.decrypt(config.get("db", "user")),
            passwd=cp.decrypt(config.get("db", "password")),
            host=config.get("db", "dev_host"),  # dev-server
            port=int(config.get("db", "port")),
            db=config.get("db", "database"),
            charset=config.get("db", "charset")
        )

except pymysql.Error as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)

# Get Cursor

def insertModelResult1(keys, scores, model):
    cur = conn.cursor()
    insertQuery1 = config.get("query", "insertQuery1")
    insertCnt = 0
    idx = 0
    for score in scores:
        key = keys[idx]
        cur.execute(insertQuery1, (key, model, round(Decimal(score), 3)))
        insertCnt = insertCnt + cur.rowcount
        idx = idx + 1
    print(insertCnt, "record inserted.")
# Get Cursor

def insertModelResult2(key, model):
    cur = conn.cursor()
    sql = config.get("query", "insertQuery")
    cur.execute(sql, key, model)
    print(cur.rowcount, "record inserted.")

def insertScoreResult(keys, scores, model):
    cur = conn.cursor()
    selectInfoQuery = config.get("query", "selectInfoQuery")
    insertScoreQuery = config.get("query", "insertScoreQuery")
    insertCnt = 0
    idx = 0
    for score in scores:
        key = keys[idx]
        cur.execute(selectInfoQuery, key)
        rows = cur.fetchall()
        for row in rows:
            cur.execute(insertScoreQuery,
                        (row[0], row[1], round(Decimal(score), 3), model))
            insertCnt = insertCnt + cur.rowcount
        idx = idx + 1

    print(insertCnt, "record inserted.")
