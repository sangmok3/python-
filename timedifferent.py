import pandas as pd
import numpy as np
from datetime import datetime

#직전 결제와의 시간차이 구하는 함수(구해서 DataFrame형태로 반환)

def time_diff(data):

    time_data = pd.DataFrame(data, columns=['아이디', '시간', '유저구분'])

    for i in range(len(time_data['시간'])):
        time_data['시간'][i] = datetime.strptime(
            time_data['시간'][i], '%Y-%m-%d %H:%M:%S')

    usr_list = set(time_data['유저구분'])
    usr_list = list(usr_list)

    data_ware = [[-1]*101 for _ in range(len(usr_list))]
    for empty in range(len(data_ware)):
        data_ware[empty][0] = 0
        data_ware[empty][1] = 0

    for user, user_num in zip(usr_list, range(len(usr_list))):
        user_name = time_data[time_data['유저구분'] == user]
        data_ware[user_num][0] = user
        for timediff in range(len(user_name['시간'])-1):
            data_ware[user_num][timediff +
                                2] = (time_data['시간'][timediff+1]-time_data['시간'][timediff]).seconds

    data_wares = pd.DataFrame(data_ware)
    return data_wares
