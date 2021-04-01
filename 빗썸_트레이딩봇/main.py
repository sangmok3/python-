import requests
import pandas as pd

#MACD 지표를 이용해 매수, 매도 조건을 결정하는 함수
#MACD곡선 : 단기 이동평균, 장기 이동평균
#Signal곡선 : n일동안의 MACD 이동평균
#기본값: 12일(단기), 26일(장기), 9일(signal)
#매수신호 : MACD 곡선이 시그널 곡선을 상향돌파
#매도신호 : MACD 곡선이 시그널 곡선을 하향돌파

# macd_short, macd_long, macd_signal = 12,26,9
df["MACD_short"] = df["종가"].rolling(macd_short).mean()
df["MACD_long"] = df["종가"].rolling(macd_long).mean()
df["MACD"] = df.apply(lambda x: (x ["MACD_short"]-x["MACD_long"]),axis=1)
df["MACD_signal"] = df["MACD"].rolling(macd_signal).mean()
df["MACD_sign"] = df.apply(lambda x: ("매수" if x["MACD"]>x["MACD_signal"]else "매도"),axis=1)

df[["종가","MACD_sign"]]


# 2021년 03월 04일~2021년 03월 05일 프로그램 생성
#made by sangmoklee
#PERMISION에러 나면 엑셀 파일이 열려있는 경우가 대부분임 닫고 다시하면 됨 
import requests
import pandas as pd
import copy
import time
import pybithumb

if input("거래를 이어서 하려면 1, 처음 시작이면 0을 입력하세요")==0:
    df_history = pd.DataFrame([],columns = ["order_status","order_st","coin_name","order_no","cash_type",
                                       "price","order_units","remain","result","order_time","finish_time"])
    df_history.to_excel("C:/Users/MS/Desktop/대박가자.xlsx")
    df_history = pd.read_excel("C:/Users/MS/Desktop/대박가자.xlsx")
else:
    df_history = pd.read_excel("C:/Users/MS/Desktop/대박가자.xlsx")

coin_name = input("원하는 코인 코드를 입력하세요") #'BTC'비트코인, 'ETH'이더리움, 'XRP' 리플
stay_persent = input("매도시 원하는 이득 비율을 입력하세요-예시 5%는 0.05, 10%는 0.1") #매수금액에서 어느정도 올라야 팔지?

# 키는 암호화 예정
import pybithumb
api_key = ""
secret_key = ""
bithumb = pybithumb.Bithumb(api_key, secret_key)

i=1

while i>0:
    #public키로 데이터 조회
    
    df_history_bid = df_history[df_history["order_status"]=="bid"]
    
    candle_period = "1M" #몇분간격으로 데이터를 scan해서 판단할지?
    juso = "https://api.bithumb.com/public/candlestick/{coin}_KRW/{t}".format(coin=coin_name, t =candle_period)

    data = requests.get(juso)
    data = data.json()
    data = data.get('data')
    df = pd.DataFrame(data) 

    df.rename(columns ={0:'time',1:'시가',2:'종가',3:'고가',4:'저가',5:'거래량'},inplace = True)

    df.sort_values("time",inplace = True) #시간순 정렬

    import time 
    df["date"]=df["time"].apply(lambda x:time.strftime('%Y-%m-%d %H:%M', time.localtime(x/1000)))

    df = df.tail(100)

    df =df.reset_index(drop=True)

    macd_short, macd_long, macd_signal = 3,7,1
    df["MACD_short"] = df["종가"].rolling(macd_short).mean()
    df["MACD_long"] = df["종가"].rolling(macd_long).mean()
    df["MACD"] = df.apply(lambda x: (x ["MACD_short"]-x["MACD_long"]),axis=1)
    df["MACD_signal"] = df["MACD"].rolling(macd_signal).mean()
    df["MACD_sign"] = df.apply(lambda x: ("매수" if x["MACD"]>x["MACD_signal"]else "매도"),axis=1)

    balance = bithumb.get_balance(coin_name)

    if balance[0]>0 and df["MACD_sign"][len(df)-1]=="매도" and float(df["시가"][len(df)-1]) > float(df_history_bid.tail(1)["price"])+int((df_history_bid.tail(1)["price"])*float(stay_persent)): #매도실행
        unit = bithumb.get_balance(coin_name)[0]
        sell_coins = unit
        price = df.loc[len(df)-1,"종가"]
        if float(price)<1000:
            price=float(price)
        else: 
            price=int(price)   
            
        order_result = bithumb.sell_limit_order(coin_name,price,sell_coins)
        
        if type(order_result) == tuple :
            if order_result == None:
                    pass
            elif order_result[0]== 'ask':
                    df_temp = pd.DataFrame({"order_status":"ask",
                                            "order_st":"매도",
                                            "coin_name":order_result[1],
                                            "order_no":order_result[2],
                                            "cash_type":order_result[3],
                                            "price":price,
                                            "order_units":buy_coins,
                                            "remain":buy_coins,
                                            "result":buy_coins*price,
                                            "order_time":pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                                            "finish_time":0
                                            },index=[0])
                    df_history = pd.concat([df_history,df_temp],ignore_index=True)
                    print("팔았다!")

        else:
            print(order_result['message'])

            
    elif balance[2]>100000 and df["MACD_sign"][len(df)-1]=="매수":
        krw = bithumb.get_balance(coin_name)[2]
        orderbook = pybithumb.get_orderbook(coin_name)
        asks = orderbook['asks']
        sell_price = asks[0]['price']
        unit = krw/(sell_price)  #내가 주문할 수 있는 최대 개수
        buy_coins = round(unit-500,4) # 빗썸 주문이 소수점 4자리까지 가능 

        price = float(df.loc[len(df)-1,"시가"])
        if price<1000:
            price=float(price)
        else: 
            price=int(price)

        order_result = bithumb.buy_limit_order(coin_name,price,buy_coins)
        if type(order_result) == tuple :
            if order_result == None:
                    pass
            elif order_result[0]== 'bid':
                    df_temp = pd.DataFrame({"order_status":"bid",
                                            "order_st":"매수",
                                            "coin_name":order_result[1],
                                            "order_no":order_result[2],
                                            "cash_type":order_result[3],
                                            "price":price,
                                            "order_units":buy_coins,
                                            "remain":buy_coins,
                                            "result":buy_coins*price,
                                            "order_time":pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                                            "finish_time":0
                                            },index=[0])
                    df_history = pd.concat([df_history,df_temp],ignore_index=True)
                    print("샀다!")

        else:
            print(order_result['message'])
#if문을 더 넣은 이유 : tuple은 거래가 성사시에 tuple형태로 나오고 만약 돈이 안맞거나 최소 주문호가가 안맞으면 dict형태로 오류를 뱉으므로 
#dict형태의 오류는 모두 pass (프린트 오류내용)/ tuple만 로그 남김 

    df_history.to_excel("C:/Users/MS/Desktop/대박가자.xlsx")
    
    print("돈 버는 중...")
    
    
    
    
    df_history_bid = 0 #메모리 초기화
    
    time.sleep(10)


# 오류패치 1. 2021년 03월 06일
#너무 잦은 매수매도(금액 이익이 거의 없음)로 인해 수수료만 많이 나감, - 존버모드 필요
#전에 매수한 금액에서 5%이상 이익이 났을때만 매도하는 조건 추가 (마지막 매수금액을 가져와서-temp로 메모리에 잠시 저장 후 반복문 마지막에 초기화)
#문제점 - 매번 루프마다 데이터 프레임으로 임시데이터(temp)를 만들게 되므로 속도 & 메모리 사용량 측면에서 비효율적임 
#시행착오 - index로 인해 매수금액 마지막꺼를 가져올때 len()로 가져올 수 없으므로 따로 매수만
#빼서 tail(1)로 뽑아냄 
##간단하지만 result에 price와 order_unit곱한 거 추가