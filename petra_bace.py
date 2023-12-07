from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm


class Colors:
    def __init__(self, num_colors=80):
        self.num_colors = num_colors
        self.color_palette = self.generate_color_palette()

    def generate_color_palette(self):
        hsv_palette = np.zeros((self.num_colors, 1, 3), dtype=np.uint8)
        hsv_palette[:, 0, 0] = np.linspace(0, 180, self.num_colors, endpoint=False)
        hsv_palette[:, :, 1:] = 255
        bgr_palette = cv2.cvtColor(hsv_palette, cv2.COLOR_HSV2BGR)
        return bgr_palette.reshape(-1, 3)

    def __call__(self, class_id):
        color = tuple(map(int, self.color_palette[class_id]))
        return color

model = YOLO(r"weights\crowd_human\plaza.pt")
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

classes = model.names
list(classes.values())[0:10]
colors=Colors()
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
yellow = (0, 255, 255)
orange = (0, 165, 255)
purple = (155, 102, 178)
white = (255,255,255)
black = (0,0,0)

triangle_coords = [(107,550), (1915,775), (1912,322)]

def draw_triangle(frame, triangle_coords):
    triangle_coords = np.array(triangle_coords, np.int32)
    cv2.drawContours(frame, [triangle_coords], 0, orange, thickness=2)
    return frame

def is_in_area(triangle_coords, obj_center):
    points = np.array(triangle_coords, np.int32)
    points = points.reshape((-1, 1, 2))
    x, y = obj_center

    # Check if the point lies inside the triangle
    result = cv2.pointPolygonTest(points, (x, y), measureDist=False)

    if result >= 0:
        return True
    else:
        return False

def draw_boxes(image, results):
  per_id = 1
  per_counter = 0
  objects_conuter = 0
  per_2=0

  for result in results[0].boxes.data.to("cpu"):
  # unpacking model output
    x1, y1, x2, y2 = int(result[0]), int(result[1]), int(result[2]), int(result[3])
    conf = result[4]
    class_id = int(result[5])
    class_name = classes[class_id]

    center = ((x1 + x2) // 2,(y1 + y2) // 2)

    if is_in_area(triangle_coords, center):
            # Counting inside the rectangle
            if class_id == per_id:
                per_counter += 1
            # Drawing inside the rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), colors(class_id), 2)
            label = f'{class_name} {conf:.2f}'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1 - h - 15), (x1 + w, y1), colors(class_id), -1)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
            # Counting outside the rectangle
            if class_id == per_id:
                per_2 += 1

            # Drawing outside the rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Use a different color (here: red)
            label = f'{class_name} {conf:.2f}'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1 - h - 15), (x1 + w, y1), (0, 0, 255), -1)  # Use a different color (here: red)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 125, 251), 2)

    # Drawing counts

  # drawing counts
  label_per = f'More: {per_counter}'
  label_per2 = f'Plaza: {per_2}'
  label_all = f'All: {objects_conuter}'
  cv2.putText(image, label_per, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, purple, 2)
  cv2.putText(image, label_per2, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1,(56, 125, 251), 2)
  #cv2.putText(image, label_all, (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, purple , 2)

  return image

# Load the video
# TODO: put valid path
cap = cv2.VideoCapture(r'C:\Users\Luka\Desktop\test videos\plaze\bacvice_split_2mins_Trim.mp4')

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(r'C:\Users\Luka\Desktop\DigDag - v8\projects\polaznici\Petra\bacvice_split_2mins_Trim.avi', fourcc, fps, (width, height))

# Loop through the frames of the video
frame_n = 0
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
      cv2.rectangle(frame, (0, 0), (250,100), (255,255,255), -1)
      frame_n += 1

      # draw rectangle on each frame
      frame = draw_triangle(frame,triangle_coords)
      label_frame = f'Frame: {frame_n}'
      cv2.putText(frame, label_frame, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, orange , 2)
      # Process the frame here
      result = model(frame, verbose=False)
      frame = draw_boxes(frame, result)
      out.write(frame)

      
      # Show the frame
      cv2.imshow('video', frame)

      # Wait for the user to press a key (optional)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    else:
      break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()