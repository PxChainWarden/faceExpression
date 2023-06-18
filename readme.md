Projedeki gerekli kütüphaneleri hızlıca yüklemek için python virualenv kullanmanızı tavsiye ederiz.

Linux için:
`pip install virtualenv`

Proje dizininde yeni bir virtual environment oluşturun:
`python -m venv venv`

Bu virtual environment'ı aktif edin:
`source venv/bin/activate`

Aşağıdaki şekilde gerekli kütüphaneleri yüklemeyi gerçekleştirin:
`pip install -r requirements.txt`

Eğitimi başlatmak için:

`python train.py --model ${model_out_file} --plot ${plot_graph_out_file} --network ${network_type}`

-m: model çıktı yolu
-p: plot grafiği çıktı yolu
-n: yapay zeka ağı tipi (VGG11 / VGG13 / VGG16 / VGG19 / resnet)

Örnek çalıştırma:

`python train.py --model output/modelvgg11.pth --plot output/plotvgg11.png -n VGG11`


Kamera üzerinden mimik tanımayı başlatmak için:


`python emotion_detection.py --network ${network_type}  --model ${model_path} --prototxt ${prototxt_path_for_face_detect} --caffemodel ${caffee_model_path_for_face_detect}`

Elimizde resnet ile eğitilmiş bir örnek bulunmakta bunu çalıştırmak için:

`python emotion_detection.py -n resnet --model output/modelnew.pth --prototxt model/deploy.prototxt.txt --caffemodel model/res10_300x300_ssd_iter_140000_fp16.caffemodel`