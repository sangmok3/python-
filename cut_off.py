import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


# 원하는 모델명과 test데이터 셋명 리스트에 입력
model_lists = ["rfc_t11.sav", "rfc_t11.sav", "rfc_t11.sav",
               "xgb_t11.sav", "xgb_t11.sav", "xgb_t11.sav"]
#model_lists =["rfc_t24.sav"]
#scale_lists =["rfc_t02_scaler.pkl"]
data_lists = ["6_day_t2.csv", "7_day_t2.csv", "8_day_t2.csv",
              "6_day_t2.csv", "7_day_t2.csv", "8_day_t2.csv"]
#data_lists = ["8_day_t2_new.csv"]


class CutOff:
  def cut_off(model_list,data_list):
    for i in range(len(model_lists)):
        model = pickle.load(open(
            '경로'
        #scaler=pickle.load(open('/home/grudeep/hg/python-machine-learning/result/'+str(scale_lists[i])+'', 'rb'))
        test = pd.read_csv(
            '경로'
        # test_y=test.iloc[:,61]
        # test_x=test.drop(['TEMP61','KEYID'],axis=1)
        test_y = test.iloc[:, -1]
        test_x = test.iloc[:, 1:-1]
        # test_x=scaler.transform(test_x)

        predict_cutoff = model.predict_proba(test_x)
        predict_cutoff = pd.DataFrame(predict_cutoff)
        result = pd.concat([predict_cutoff[1], test_y], axis=1)
        result.columns = ['cut', 'real']
        result1 = pd.concat([predict_cutoff[0], test_y], axis=1)
        result1.columns = ['cut', 'real']
        fraud = result[result['real'] == 1]
        normal = result[result['real'] == 0]
        f_normal = result1[result1['real'] == 0]

        plt.figure(figsize=(8, 8))
        plt.plot(normal['cut'], marker='o', ms=3, linestyle='')
        plt.plot(fraud['cut'], marker='o', ms=3, linestyle='')
        plt.grid(color='lightgray')
        plt.legend('center right', frameon=True, labels=[
                   'Normal', 'Fraud'], fontsize=8)
        plt.title(''+str(model_lists[i])+'_Cut-off', fontsize=15)
        plt.xlabel('Number of Samples')
        plt.ylabel('Prediction score')

        # fraud 카운트를 위한 변수
        fraud_c1 = len(fraud[round(fraud["cut"], 1) >= 0.1])
        fraud_c2 = len(fraud[round(fraud["cut"], 1) >= 0.2])
        fraud_c3 = len(fraud[round(fraud["cut"], 1) >= 0.3])
        fraud_c4 = len(fraud[round(fraud["cut"], 1) >= 0.4])
        fraud_c5 = len(fraud[round(fraud["cut"], 1) >= 0.5])
        fraud_c6 = len(fraud[round(fraud["cut"], 1) >= 0.6])
        fraud_c7 = len(fraud[round(fraud["cut"], 1) >= 0.7])
        fraud_c8 = len(fraud[round(fraud["cut"], 1) >= 0.8])
        fraud_c9 = len(fraud[round(fraud["cut"], 1) >= 0.9])
        fraud_cnt = fraud_c1, fraud_c2, fraud_c3, fraud_c4, fraud_c5, fraud_c6, fraud_c7, fraud_c8, fraud_c9

        # normal 카운트를 위한 변수
        normal_c1 = len(normal[round(normal["cut"], 1) >= 0.1])
        normal_c2 = len(normal[round(normal["cut"], 1) >= 0.2])
        normal_c3 = len(normal[round(normal["cut"], 1) >= 0.3])
        normal_c4 = len(normal[round(normal["cut"], 1) >= 0.4])
        normal_c5 = len(normal[round(normal["cut"], 1) >= 0.5])
        normal_c6 = len(normal[round(normal["cut"], 1) >= 0.6])
        normal_c7 = len(normal[round(normal["cut"], 1) >= 0.7])
        normal_c8 = len(normal[round(normal["cut"], 1) >= 0.8])
        normal_c9 = len(normal[round(normal["cut"], 1) >= 0.9])
        normal_cnt = normal_c1, normal_c2, normal_c3, normal_c4, normal_c5, normal_c6, normal_c7, normal_c8, normal_c9

        # 동적변수 생성(컷오프 개수만큼 0.1~0.9)
        fraud_ea = 9
        for i, m, m1 in zip(range(1, fraud_ea+1), fraud_cnt, normal_cnt):
            globals()['Precision_{}'.format(i)] = round(
                m/(m+m1+0.00001)*100, 1)
            globals()['Recall_{}'.format(i)] = round((m/(len(fraud)+m))*100, 1)

        Precision_cnt = Precision_1, Precision_2, Precision_3, Precision_4, Precision_5, Precision_6, Precision_7, Precision_8, Precision_9
        Recall_cnt = Recall_1, Recall_2, Recall_3, Recall_4, Recall_5, Recall_6, Recall_7, Recall_8, Recall_9

        # 컷오프 이미지 출력
        plt.show()

        # 결과값 출력
        for n, z, z1, p, r in zip(range(1, 10), fraud_cnt, normal_cnt, Precision_cnt, Recall_cnt):
            print(f"precision_0."+str(n) + "   :\t"+str(p)+"%")
            print(f"recall_0."+str(n) + "      :\t"+str(r)+"%")
            print(f"fraud_count_0."+str(n) + " :\t"+str(z)+"")
            print(f"normal_count_0."+str(n) + ":\t"+str(z1)+"")
            print("----------------------------------------")
