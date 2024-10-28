import cv2
import math
import json
import torch
import numpy as np
from collections import deque
from mmpose.apis.inferencers import MMPoseInferencer
from module.Loader import TSSTG

def init_args(video=None):
    if video is None:
        video = './static/demo/clip_short.mp4'
    args = {
        'video': video,
        'checkpoint': './checkpoint/falldown_checkpoint.pth',
        'threshhold': 0.8,
        'frame_step': 14,
        'device': 'cuda:0',
        'extract_args': {
            'init_args': {
                'pose2d': 'rtmo', 
                'pose2d_weights': './checkpoint/rtmo_checkpoint.pth', 
                'scope': 'mmpose', 
                'device': 'cuda:0', 
                'det_model': None, 
                'det_weights': None, 
                'det_cat_ids': 0, 
                'pose3d': None, 
                'pose3d_weights': None, 
                'show_progress': False
            },
            'call_args': {
                'inputs': video, 
                'show': False, 
                'draw_bbox': True, 
                'draw_heatmap': False, 
                'bbox_thr': 0.5, 
                'nms_thr': 0.65, 
                'pose_based_nms': True, 
                'kpt_thr': 0.3, 
                'tracking_thr': 0.3, 
                'use_oks_tracking': False, 
                'disable_norm_pose_2d': False, 
                'disable_rebase_keypoint': False, 
                'num_instances': 1, 
                'radius': 3, 
                'thickness': 1, 
                'skeleton_style': 'openpose', 
                'black_background': False, 
                'vis_out_dir': '', 
                'pred_out_dir': '', 
                'vis-out-dir': './'
            }
        }
    }
    return args

def pre_processing(skeletons, frame_step):
    skeletons = deque(skeletons, maxlen=frame_step)
    for i, sk in enumerate(skeletons):
        if i == frame_step:
            break
        indices_14 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]        
        skeletons[i] = sk[indices_14]

    return np.array(skeletons)

async def inference(video=None):
    yield f"data: {json.dumps({'message': f'Inference Start'})}\n\n"
    yield f"data: {json.dumps({'message': f'Initalizing Arguments...'})}\n\n"
    args = init_args(video)
    _init_args = args['extract_args']['init_args']
    _call_args = args['extract_args']['call_args']
    yield f"data: {json.dumps({'message': f'Initalizing Arguments... Done!'})}\n\n"

    yield f"data: {json.dumps({'message': f'Initalizing Extract Model...'})}\n\n"
    inferencer = MMPoseInferencer(**_init_args)
    yield f"data: {json.dumps({'message': f'Initalizing Extract Model... Done!'})}\n\n"
    yield f"data: {json.dumps({'message': f'Initalizing Falldown Model...'})}\n\n"
    model = TSSTG(args['checkpoint'], args['device'])
    yield f"data: {json.dumps({'message': f'Initalizing Falldown Model... Done!'})}\n\n"

    cap = cv2.VideoCapture(_call_args['inputs'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    falldown_result = []
    yield f"data: {json.dumps({'message': f'Extracting Video and Detecting Falldown...'})}\n\n"
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                running_time = format(frame_index / fps, '.2f')
                skeletons = []
                progress = math.ceil(frame_index / frame_count * 100)
                
                temp_call_args = _call_args
                temp_call_args['inputs'] = frame
                results = inferencer(**temp_call_args)
                for result in results:
                    pred = result['predictions'][0]
                    pred.sort(key = lambda x: x['bbox'][0][0])

                    for p in pred:
                        keypoints = p['keypoints']
                        keypoints_scores = p['keypoint_scores']
                        skeletons.append([a + [b] for a, b in zip(keypoints, keypoints_scores)])

                skeletons = [skeletons[0] for _ in range(args['frame_step'])]
                skeletons = np.array(skeletons, dtype=np.float32)
                skeletons = pre_processing(skeletons=skeletons, frame_step=args['frame_step'])
                out = model.predict(skeletons, frame_size)
                label = model.class_names[out[0].argmax()]
                confidence = out[0][1]
                if label == 'Fall Down' and confidence > args['threshhold']:
                    falldown_result.append((running_time, label, confidence))
                yield f"data: {json.dumps({'progress': f'{progress}'})}\n\n"
                yield f"data: {json.dumps({'message': f'{running_time}s, action: {label}, {confidence}'})}\n\n"
                frame_index += 1
            else:
                yield f"data: {json.dumps({'progress': f'{100}'})}\n\n"
                break
    cap.release()
    yield f"data: {json.dumps({'message': f'Extracting Video and Detecting Falldown... Done'})}\n\n"
    yield f"data: {json.dumps({'message': f'Inference Result: {falldown_result}'})}\n\n"
    yield f"data: {json.dumps({'message': f'Visualize Result'})}\n\n"
    if len(falldown_result) == 0:
        yield f"data: {json.dumps({'message': f'No falldown were detected.'})}\n\n"
    else:
        for row in falldown_result:
            running_time, label, confidence = row
            yield f"data: {json.dumps({'message': f'[{running_time}s] {label}: {confidence}%'})}\n\n"