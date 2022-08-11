cd /d "D:\research\osr"
call C:\Users\tuan\AppData\Local\Continuum\anaconda3\Scripts\activate.bat C:\Users\tuan\AppData\Local\Continuum\anaconda3\envs\osr
timeout 5

start /wait python consts_updater.py -d 0
timeout 5
start /wait python main.py
timeout 5

start /wait python consts_updater.py -d 1
timeout 5
start /wait python main.py
timeout 5

start /wait python consts_updater.py -d 2
timeout 5
start /wait python main.py
timeout 5

start /wait python consts_updater.py -d 3
timeout 5
start /wait python main.py
timeout 5

start /wait python consts_updater.py -d 4
timeout 5
start /wait python main.py
timeout 5

start /wait python consts_updater.py -d 5
timeout 5
start /wait python main.py
timeout 5

start /wait python consts_updater.py -d 6
timeout 5
start /wait python main.py
timeout 5

start /wait python consts_updater.py -d 7
timeout 5
start /wait python main.py
timeout 5

start /wait python consts_updater.py -d 8
timeout 5
start /wait python main.py
timeout 5

start /wait python consts_updater.py -d 9
timeout 5
start /wait python main.py
timeout 5

start /wait python consts_updater.py -d 10
timeout 5
start /wait python main.py
timeout 5

start /wait python consts_updater.py -d 11
timeout 5
start /wait python main.py
timeout 5

exit
