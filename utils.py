import cv2
from ultralytics import YOLO
import random
import gradio as gr
from tqdm import tqdm

class yolo_model():
    def __init__(self, model_name: str):
        """
        Initialize the YOLO-World model

        Args:
            model_name (str): The name of the model file.
        """

        # Initialize a YOLO-World model
        self.model = YOLO(model_name)

    def load(self, model_name: str):
        """
        Load the YOLO model

        Args:
            model_to_load (str): The name of the model file.
        """

        try:
            # Load the model
            self.model = YOLO(model_name)

        except Exception as e:
            print(e)

    # Define a function to process a video
    def process(self, video_path: str, prompt: str, confidence: float, iou: float, progress=gr.Progress(track_tqdm=True)
                ) -> str:
        """
        Process a video with YOLO-World

        Args:
            video_path (str): The input video path.
            confidence (float): The confidence threshold.
            iou (float): The IoU threshold.

        Returns:
            str: The output video path.

        """

        try:

            # create a list of classes based on prompt, each class is separated by a comma
            classes = prompt.split(",") if prompt else None

            # Define the colors for each class
            rgb_colors = [(random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255)) for _ in range(len(classes))]

            # Define custom classes
            self.model.set_classes(classes)

            # Set confidence and IoU thresholds
            self.model.conf = confidence
            self.model.iou = iou

            # Open the video file
            video_capture = cv2.VideoCapture(video_path)

            # Get the video properties
            frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video_capture.get(cv2.CAP_PROP_FPS))
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            # Define the output video path
            output_video_path = 'output.mp4'

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video_writer = cv2.VideoWriter(
                output_video_path, fourcc, fps, (frame_width, frame_height), isColor=True)

            # Process each frame in the video
            for _ in tqdm(range(n_frames), desc="Processing video", file=progress):
                
                ret, frame = video_capture.read()
                if not ret:
                    break  # Break the loop when no frames are left

                # Run inference to detect your custom classes
                results = self.model.predict(frame)

                if len(results) > 0:
                    # Extract the bounding boxes and class names
                    boxes = results[0].boxes.cpu().numpy().data
                    class_names = self.model.names  # Load class names if you need them

                    for box in boxes:
                        x1, y1, x2, y2, conf, class_id = box.tolist()  # Convert normalized coordinates

                        # convert to int
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        label = f'{class_names[class_id]}: {conf:.2f}'

                        # Draw bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2),
                                      rgb_colors[int(class_id)], 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb_colors[int(class_id)], 2)

                # Write the grayscale frame to the output video
                video_writer.write(frame)

            # Release resources
            video_capture.release()
            video_writer.release()

            # Return the output video path
            return output_video_path

        except Exception as e:
            print(e)
            return None
