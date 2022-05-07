# Shanghai COVID-19 2022

## Desciption
This script downloads daily reports from [Shanghai Health Commission](https://wsjkw.sh.gov.cn/xwfb/index.html)  
And then cache urls and cases in current directory.  
And then generate address reports for each district (LibreOffice Calc | .ods)  

此脚本从[上海卫健委官网](https://wsjkw.sh.gov.cn/xwfb/index.html)下载每日通报  
然后在当前目录中缓存URL和病例数据  
然后按行政区保存地址分析结果 (LibreOffice Calc | .ods)  


## Usage
```cmd
py ShCase.py --help
usage: ShCase.py [-h] [-s STARTDATE] [-e ENDDATE] [-c {url,case,html,addr,log,all}] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Shanghai COVID-19 2022

options:
  -h, --help            show this help message and exit
  -s STARTDATE, --startdate STARTDATE
                        The Start Date - format YYYY-MM-DD
  -e ENDDATE, --enddate ENDDATE
                        The End Date format YYYY-MM-DD (Inclusive)
  -c {url,case,html,addr,log,all}, --clean {url,case,html,addr,log,all}
                        Clean local data
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level
```


## Requirements
* Python 3.8+ (3.10 tested only)
* requests
* Numpy
* Pandas
* lxml
* odfpy | openpyxl | xlsxwriter


## Files

| Path             | Description                               |
|------------------|-------------------------------------------|
| urls(.pkl .txt)  | URLs in pickle and pprint formatted data  |
| cases(.pkl .txt) | Cases in pickle and pprint formatted data |
| html             | Downloaded html pages                     |
| reports          | Spreadsheet reports                       |
| ShCase.py        | Main script                               |
| ShCase.log       | Logging file                              |
