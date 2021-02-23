# car-classifier
Проект по классификации автомашин

## Описание
- **cars_classifier_ntbk.ipynb** - обучение модели классификации марки и модели
- **car_classifier.py** - итоговый класс для предсказания марки, модели и цвета
- **model/**:
    - **model_classifier.pth** - обученная модель классификатора марки и модели
    - **idx_to_class.json** - названия классов (марка+модель) автомобилей

## Запуск скрипта
```python car_classifier.py --model model_folder```  

В папке `model_folder` должны быть файлы `idx_to_class.json` и `model_classifier.pth`  

После запуска скрипт предложит ввести путь до файла с изображением:  
```Enter path to image (type "exit" to quit): image/000001.jpg```

После ввода скрипт выдаст название марки и модели автомобиля, а также ее цвет:  
``` -> AM General Hummer SUV 2000 (yellow)```  

Скрипт будет работать до тех пор, пока не будет введено "exit"

## Данные, использованные в проекте:
1. Cars Dataset: [описание](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), [данные](http://imagenet.stanford.edu/internal/car196/car_ims.tgz) и [разметка](http://imagenet.stanford.edu/internal/car196/cars_annos.mat).
3. Vehicle Color Dataset: [описание](http://cloud.eic.hust.edu.cn:8071/~pchen/project.html), [данные](http://cloud.eic.hust.edu.cn:8071/~pchen/color.rar)
