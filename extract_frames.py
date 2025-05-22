import cv2
import os

class VideoFrameExtractor:
    def __init__(self, video_path, output_folder='frames', interval=2):
        """
        Initialize the VideoFrameExtractor.

        Parameters:
        video_path (str): Path to the video file.
        output_folder (str): Directory where extracted frames will be saved.
        interval (int): Interval in seconds between frames to be extracted.
        """
        self.video_path = video_path
        self.output_folder = output_folder+'_'+video_path.split('.')[0] # Change 1
        self.interval = interval
        self.timestamps = []

        # Create the frames directory if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
        frame_interval = int(frame_rate * self.interval)

        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
# Cut frames
            if frame_count % frame_interval == 0:
                timestamp = frame_count / frame_rate
                self.timestamps.append(timestamp)
                frame_filename = f'{self.output_folder}/frame_{frame_count}.jpg'
                cv2.imwrite(frame_filename, frame)
                print(f'Saved frame at {timestamp:.2f} seconds')
# Cut frames
            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

    def get_timestamps(self):
        return self.timestamps
    
    def get_output_folder(self):
        return self.output_folder  

# Example usage
# extractor = VideoFrameExtractor('path_to_video.mp4')
# extractor.extract_frames()
# print('Timestamps:', extractor.get_timestamps())
