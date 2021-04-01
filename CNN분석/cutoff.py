def cut_off(_) : 
    import matplotlib.pyplot as plt

    fraud  = _[_[_.columns[-1]]==1]
    normal = _[_[_.columns[-1]]==0]

    plt.figure(figsize=(8,8))
    plt.plot(normal['cut_off'], marker='o',ms=3, linestyle='')
    plt.plot(fraud['cut_off'], marker='o',ms=3, linestyle='')
    plt.grid(color = 'lightgray')
    plt.legend('center right',frameon = True, labels =['Normal','Fraud'], fontsize = 8)
    plt.title('deep_model_Cut-off', fontsize=15)
    plt.xlabel('Number of Samples')
    plt.ylabel('Predict score')

    #fraud 카운트를 위한 변수
    fraud_c1= len(fraud[round(fraud["cut_off"],1)>0.099999999]);fraud_c2=len(fraud[round(fraud["cut_off"],1)>0.199999999]);fraud_c3=len(fraud[round(fraud["cut_off"],1)>0.299999999]);fraud_c4=len(fraud[round(fraud["cut_off"],1)>0.399999999]);fraud_c5=len(fraud[round(fraud["cut_off"],1)>0.499999999])
    fraud_c6= len(fraud[round(fraud["cut_off"],1)>0.599999999]);fraud_c7=len(fraud[round(fraud["cut_off"],1)>0.699999999]);fraud_c8=len(fraud[round(fraud["cut_off"],1)>0.799999999]);fraud_c9=len(fraud[round(fraud["cut_off"],1)>0.899999999])
    fraud_cnt= fraud_c1,fraud_c2,fraud_c3,fraud_c4,fraud_c5,fraud_c6,fraud_c7,fraud_c8,fraud_c9

    #normal 카운트를 위한 변수
    normal_c1= len(normal[round(normal["cut_off"],1)>0.099999999]);normal_c2=len(normal[round(normal["cut_off"],1)>0.199999999]);normal_c3=len(normal[round(normal["cut_off"],1)>0.299999999]);normal_c4=len(normal[round(normal["cut_off"],1)>0.399999999]);normal_c5=len(normal[round(normal["cut_off"],1)>0.499999999])
    normal_c6= len(normal[round(normal["cut_off"],1)>0.599999999]);normal_c7=len(normal[round(normal["cut_off"],1)>0.699999999]);normal_c8=len(normal[round(normal["cut_off"],1)>0.799999999]);normal_c9=len(normal[round(normal["cut_off"],1)>0.899999999])
    normal_cnt= normal_c1,normal_c2,normal_c3,normal_c4,normal_c5,normal_c6,normal_c7,normal_c8,normal_c9


    #동적변수 생성(컷오프 개수만큼 0.1~0.9)
    fraud_ea = 9
    for i,m,m1 in zip(range(1,fraud_ea+1),fraud_cnt,normal_cnt):
        globals()['Precision_{}'.format(i)]= round(m/(m+m1+0.00001)*100,1)
        globals()['Recall_{}'.format(i)]= round((m/(len(fraud)+m))*100,1)

    Precision_cnt = Precision_1,Precision_2,Precision_3,Precision_4,Precision_5,Precision_6,Precision_7,Precision_8,Precision_9
    Recall_cnt = Recall_1,Recall_2,Recall_3,Recall_4,Recall_5,Recall_6,Recall_7,Recall_8,Recall_9

    #컷오프 이미지 출력
    plt.show()

    #결과값 출력
    for n,z,z1,p,r in zip(range(1,10),fraud_cnt,normal_cnt,Precision_cnt,Recall_cnt):            
        print(f"precision_0."+str(n)+ "   :\t"+str(p)+"%")
        print(f"recall_0."+str(n)+ "      :\t"+str(r)+"%")
        print(f"fraud_count_0."+str(n)+ " :\t"+str(z)+"")
        print(f"normal_count_0."+str(n)+ ":\t"+str(z1)+"")
        print("----------------------------------------")