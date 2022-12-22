
## demos on ImageNet 1K ##

    # orange
python main.py --resize_y --config confs/inet256.yml --path_y data/datasets/gts/inet256/orange.png --class 950 --deg "sr_averagepooling" --scale 4 -i orange

    # brown bear
python main.py --resize_y --config confs/inet256.yml --path_y data/datasets/gts/inet256/bear.png --class 294 --deg "sr_averagepooling" --scale 4 -i bear

    # flamingo
python main.py --resize_y --config confs/inet256.yml --path_y data/datasets/gts/inet256/flamingo.png --class 130 --deg "sr_averagepooling" --scale 2 -i flamingo

    # kimono
python main.py --resize_y --config confs/inet256.yml --path_y data/datasets/gts/inet256/kimono.png --class 614 --deg "sr_averagepooling" --scale 2 -i kimono

    # zebra 
python main.py --resize_y --config confs/inet256.yml --path_y data/datasets/gts/inet256/zebra.png --class 340 --deg "sr_averagepooling" --scale 4 -i zebra
