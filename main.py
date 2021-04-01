# configparser => ini파일 읽어오는데 사용 - ini파일(config.properties)은 섹션[data],키:traindata,값:none으로 이루어짐
import configparser as cf
import sys
import json
import time
import argparse
import functions as fc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help="Properties path", required=True)
    args = parser.parse_args()
    print(args.p)
    config = cf.ConfigParser()
    config.read(args.p, encoding='utf-8')

    print(config.get("path", "jsonPath"))
    models = json.load(open(config.get("path", "jsonPath")))

    for model in models:
        option = model["option"]
        config.set("data", "trainData", model["trainData"])
        config.set("data", "testData", model["testData"])
        config.set("data", "fileName", model["fileName"])
        config.set("data", "scaleMethod", model["scaleMethod"])
        fc.sortOption(option, config)
        time.sleep(10)
