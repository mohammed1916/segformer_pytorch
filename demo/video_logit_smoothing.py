# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import cv2
import torch
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model
from mmseg.apis.inference import show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('video', help='Video file or webcam id')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--palette', default='cityscapes', help='Color palette')
    parser.add_argument('--show', action='store_true', help='Whether to show result')
    parser.add_argument('--show-wait-time', default=1, type=int, help='Wait time after imshow')
    parser.add_argument('--output-file', default=None, type=str, help='Output video file path')
    parser.add_argument('--output-fourcc', default='MJPG', type=str, help='Fourcc of output video')
    parser.add_argument('--output-fps', default=-1, type=int, help='FPS of output video')
    parser.add_argument('--output-height', default=-1, type=int, help='Frame height')
    parser.add_argument('--output-width', default=-1, type=int, help='Frame width')
    parser.add_argument('--opacity', type=float, default=0.5, help='Opacity of painted map')
    parser.add_argument('--smooth-alpha', type=float, default=0.6,
                        help='Weight for previous logits in smoothing (0-1)')
    args = parser.parse_args()

    assert args.show or args.output_file, 'At least one output should be enabled.'

    # build the model
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # open video
    if args.video.isdigit():
        args.video = int(args.video)
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened()
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    # output video init
    writer = None
    if args.output_file is not None:
        fourcc = cv2.VideoWriter_fourcc(*args.output_fourcc)
        output_fps = args.output_fps if args.output_fps > 0 else input_fps
        output_height = args.output_height if args.output_height > 0 else input_height
        output_width = args.output_width if args.output_width > 0 else input_width
        writer = cv2.VideoWriter(args.output_file, fourcc, output_fps,
                                 (output_width, output_height), True)

    prev_logits = None  # store previous frame logits

    try:
        while True:
            flag, frame = cap.read()
            if not flag:
                break

            # run inference
            result = inference_model(model, frame)

            # get logits tensor
            logits = result.pred_sem_seg.data[0].float()  # [num_classes, H, W]

            # apply temporal smoothing
            if prev_logits is not None:
                logits = (1 - args.smooth_alpha) * logits + args.smooth_alpha * prev_logits
            prev_logits = logits.clone()

            # replace result logits with smoothed version
            result.pred_sem_seg.data[0] = logits

            # visualize
            draw_img = show_result_pyplot(model, frame, result)

            if args.show:
                cv2.imshow('video_demo', draw_img)
                cv2.waitKey(args.show_wait_time)
            if writer:
                if draw_img.shape[0] != output_height or draw_img.shape[1] != output_width:
                    draw_img = cv2.resize(draw_img, (output_width, output_height))
                writer.write(draw_img)
    finally:
        if writer:
            writer.release()
        cap.release()


if __name__ == '__main__':
    main()
