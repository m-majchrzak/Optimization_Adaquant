# Optimization_Adaquant
ToDo:
- ~~load dataset: ImageNet (Agata)~~
- ~~model - ResNest18~~
- ~~create calibration dataset~~
- clean code (main.ipynb) (Martyna) - DONE

ToDo (Code):
- podmienić w main train_data na calibration data?
- stworzyć cached_qinput jako zmienną globalną  zobaczyć, czy wtedy stworzy się jakiś cached_input_output. Spróbować odpalić main.ipynb do końca
- zmienić metrykę z mse na accuracy
- spróbować zrobić tak żeby accuracy działało z topk=(1,5) (output musi mieć conajmniej 5 najbardziej prawdopodobnych klas)
- zobaczyć jakie accuracy ma ResNet_imagenet z ich implementacji
- czy używamy ich systemu logowania (setup_logging i ResultsLog w log.py)?


Link do imagenet https://wutwaw-my.sharepoint.com/:f:/g/personal/01151431_pw_edu_pl/EkhZA9bny-tAk3JgIQuuWj4BM_qYwctETXdPUOS2K7om9Q?e=Nlxy1f
