import cv2

def mp4_to_frames(video='JumpingJacks.mp4'):
  vidcap = cv2.VideoCapture(video)
  success,image = vidcap.read()
  print(success)
  count = 0
  while success:
    cv2.imwrite("frames/frame%d.png" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()
    print(f"Processing frame {count}")
    count += 1
  print("Finished processing video")
