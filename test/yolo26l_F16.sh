../build/yolo_ggml \
	-m ../data/gguf/yolo26l_F16.gguf \
        -i ../data/img/ancelotti_zidane_2014.tga \
        -o ../data/out/output__L_F16.tga \
        -l ../data/coco/ms_coco_classnames.txt \
	-b 1024
