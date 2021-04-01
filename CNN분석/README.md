```
The first version of this document was written by
@author: sangmoklee
@date: 2021-MAR
```

## **Objectives**
* Analyze data to find the insights for detect charge-backs
* Develop model for detect charge-backs


## **Folder tree**
```
---- folder
- file

 python_programming
    | 
    |---- CNN_시계열 분석
    |
    |- main.py                      : 차지백 유저 구분 model_run
    |
    |- function.py                  : 데이터 Null값 개수 확인, 4분위수 확인, 평균 확인 등  
    |
    |- seperate.py                  : 데이터 X,Y분류 / CNN형태로 데이터 shape변환 함수
    |
    |- model.py                     : DNN, CNN모델(Conv2D, Conv3D)
    |
    |- function.py                  : 데이터 Null값 개수 확인, 4분위수 확인, 평균 확인 / 이진분류 함수 등  
    |
    |- package.py                   : 필요한 package불러오기(함수 불러와서 package.pd.DataFrame()식으로 사용 가능)
    |
    |- cutoff.py                    : 컷오프별 결과값 정리, Plot이미지 생성
    |
    |----
```

## **How to use**

* **데이터 프로파일링**
    * 데이터 분류: X,Y함수를 통해 8:2로 구분(Y는 맨 마지막 열로 고정시켜서 데이터 추출)
    * 데이터 기본정보 : 결측치 확인,평균,MINMAX 확인 등


* **적용 가능한 알고리즘**
    * Deep Learning
        * Deep Neural Network
        * Convolutional Neural Network (2D, 3D)
        * Long Short Term Memory (To Be Updated)
        * Autoencoder (To Be Updated)

