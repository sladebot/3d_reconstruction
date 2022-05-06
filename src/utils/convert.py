import cv2


def mp4_to_frames(video, output_dir, target_fps=24):
    vidcap = cv2.VideoCapture(video)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    # print(f"fps: {fps}, Target FPS: {target_fps}, Divider: {fps // target_fps}")
    success, image = vidcap.read()
    count = 0
    while success:
        if count % (fps // target_fps) == 0:
            print(f"writing frame to: {output_dir}/frame%d.png" % count)
            cv2.imwrite(f"{output_dir}/frame%d.png" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
    print("Finished preprocessing video")


# mp4_to_frames("data/FigureSkater.mp4", output_dir="data/frames/FigureSkater")
