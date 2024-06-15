import cv2
import supervision as sv
from ultralytics.engine.model import Model


def YOLO_annotateImage(model: Model, image:cv2.typing.MatLike):
    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    sv.plot_image(annotated_image)
    
    
def YOLO_annotateVideo(model: Model, video_path:str):
    cap = cv2.VideoCapture(video_path) 
    
    if (cap.isOpened()== False): 
        print("Error opening video file")
        
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
  
    # Read until video is completed 
    while(cap.isOpened()): 
    # Capture frame-by-frame 
        ret, frame = cap.read() 
        if not ret: break
        
        # Display the resulting frame 
        results = model(source=frame, conf=0.25)[0]
        detections = sv.Detections.from_ultralytics(results)

        annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        # out.write(annotated_image)
        cv2.imshow('Frame', frame)
        
        # Press Q on keyboard to exit 
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
        
    # When everything done, release 
    # the video capture object 
    cap.release() 
    # out.release()
    
    # Closes all the frames 
    cv2.destroyAllWindows()
    