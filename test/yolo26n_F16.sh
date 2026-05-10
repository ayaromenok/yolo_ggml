../build/yolo_ggml \
	-m ../data/gguf/yolo26n_F16.gguf \
        -i ../data/img/ancelotti_zidane_2014.tga \
        -o ../data/out/output__N_F16.tga \
        -l ../data/coco/ms_coco_classnames.txt \
	-b 10240
