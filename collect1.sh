#!/bin/bash

bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_alllimited/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_alllimited/faster_rcnn_r50_oln_fpn/latest.pth train 4 --num-images 10000 --score-thresh 0.05
bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_alllimited/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_alllimited/faster_rcnn_r50_oln_fpn/latest.pth val 4 --score-thresh 0.05
bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_alllimited/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_alllimited/faster_rcnn_r50_oln_fpn/latest.pth test 4


bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_vehicles/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_vehicles/faster_rcnn_r50_oln_fpn/latest.pth train 4 --num-images 10000 --score-thresh 0.05
bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_vehicles/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_vehicles/faster_rcnn_r50_oln_fpn/latest.pth val 4 --score-thresh 0.05
bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_vehicles/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_vehicles/faster_rcnn_r50_oln_fpn/latest.pth test 4


bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_planes/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_planes/faster_rcnn_r50_oln_fpn/latest.pth train 4 --num-images 10000 --score-thresh 0.05
bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_planes/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_planes/faster_rcnn_r50_oln_fpn/latest.pth val 4 --score-thresh 0.05
bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_planes/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_planes/faster_rcnn_r50_oln_fpn/latest.pth test 4


bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_ships/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_ships/faster_rcnn_r50_oln_fpn/latest.pth train 4 --num-images 10000 --score-thresh 0.05
bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_ships/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_ships/faster_rcnn_r50_oln_fpn/latest.pth val 4 --score-thresh 0.05
bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_ships/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_ships/faster_rcnn_r50_oln_fpn/latest.pth test 4


bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_smallcar/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_smallcar/faster_rcnn_r50_oln_fpn/latest.pth train 4 --num-images 10000 --score-thresh 0.05
bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_smallcar/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_smallcar/faster_rcnn_r50_oln_fpn/latest.pth val 4 --score-thresh 0.05
bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_smallcar/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_smallcar/faster_rcnn_r50_oln_fpn/latest.pth test 4


bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_all75/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_all75/faster_rcnn_r50_oln_fpn/latest.pth train 4 --num-images 10000 --score-thresh 0.05
bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_all75/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_all75/faster_rcnn_r50_oln_fpn/latest.pth val 4 --score-thresh 0.05
bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_all75/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_all75/faster_rcnn_r50_oln_fpn/latest.pth test 4


bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_all50/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_all50/faster_rcnn_r50_oln_fpn/latest.pth train 4 --num-images 10000 --score-thresh 0.05
bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_all50/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_all50/faster_rcnn_r50_oln_fpn/latest.pth val 4 --score-thresh 0.05
bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_all50/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_all50/faster_rcnn_r50_oln_fpn/latest.pth test 4


bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_all25/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_all25/faster_rcnn_r50_oln_fpn/latest.pth train 4 --num-images 10000 --score-thresh 0.05
bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_all25/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_all25/faster_rcnn_r50_oln_fpn/latest.pth val 4 --score-thresh 0.05
bash tools/dist_collect_osodd_preds.sh configs/openset/fair1m_all25/faster_rcnn_r50_oln_fpn.py out/openset/fair1m_all25/faster_rcnn_r50_oln_fpn/latest.pth test 4


