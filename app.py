# Flask version of Streamlit video-merging app
from flask import Flask, request, jsonify, send_file, render_template_string
import os
import tempfile
import moviepy.editor as mpy
import random
from extract_frames import VideoFrameExtractor
from get_inference import FramePredictor
from emo_trim import select_emotion

app = Flask(__name__)

HTML_FORM = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Emotion-Based Video Merger</title>
  <style>
    body {
      background-color: #0e1117;
      color: #f0f2f6;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      margin: 0;
    }
    .container {
      background-color: #161a23;
      padding: 40px;
      border-radius: 12px;
      box-shadow: 0 0 20px rgba(255, 255, 255, 0.05);
      width: 90%;
      max-width: 600px;
    }
    h2 {
      text-align: center;
      font-size: 1.8rem;
      margin-bottom: 20px;
    }
    label {
      display: block;
      margin-top: 20px;
      font-size: 1rem;
    }
    select, input[type="file"], button {
      width: 100%;
      padding: 12px;
      margin-top: 8px;
      border-radius: 8px;
      border: none;
      font-size: 1rem;
    }
    select, input[type="file"] {
      background-color: #2c313c;
      color: #f0f2f6;
    }
    button {
      background-color: #2f80ed;
      color: white;
      font-weight: bold;
      cursor: pointer;
      margin-top: 24px;
    }
    button:hover {
      background-color: #2563eb;
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>Emotion-Based Video Merger</h2>
    <form method="post" enctype="multipart/form-data">
      <label for="emotion">Select a preferred emotion:</label>
      <select name="emotion" id="emotion">
        <option value="sad">sad</option>
        <option value="happy">happy</option>
        <option value="fear">fear</option>
        <option value="surprise">surprise</option>
        <option value="neutral">neutral</option>
        <option value="disgust">disgust</option>
        <option value="angry">angry</option>
      </select>

      <label for="videos">Upload 2 to 4 video files:</label>
      <input type="file" name="videos" id="videos" multiple accept="video/mp4,video/x-m4v,video/*">

      <button type="submit">Process Videos</button>
    </form>
  </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def upload_videos():
    if request.method == 'POST':
        emotion = request.form['emotion']
        files = request.files.getlist("videos")

        if not (2 <= len(files) <= 4):
            return "Please upload between 2 and 4 video files.", 400

        session_dir = tempfile.mkdtemp()
        video_paths = []

        for vid in files:
            path = os.path.join(session_dir, vid.filename)
            vid.save(path)
            video_paths.append(path)

        # 1. Frame extraction
        timestamps_list = []
        output_folders = []
        for vp in video_paths:
            extractor = VideoFrameExtractor(vp)
            extractor.extract_frames()
            timestamps_list.append(extractor.get_timestamps())
            output_folders.append(extractor.get_output_folder())

        # 2. Emotion inference
        predictions = []
        for out_folder in output_folders:
            predictor = FramePredictor(out_folder)  # using mock predictor
            predictions.append(predictor.predict_frames())

        # 3. Select emotion timestamps
        filtered = []
        for preds, stamps in zip(predictions, timestamps_list):
            filtered.append(select_emotion(preds, stamps, emotion))

        # 4. Merge videos
        def merge_videos(video_paths_timelines):
            entries = []
            for path, stamps in video_paths_timelines.items():
                for t in stamps:
                    entries.append((t, path))
            entries.sort(key=lambda x: x[0])
            unique = []
            i = 0
            while i < len(entries):
                ts, vid = entries[i]
                duplicates = [(ts, vid)]
                j = i + 1
                while j < len(entries) and entries[j][0] == ts:
                    duplicates.append(entries[j]); j += 1
                unique.append(random.choice(duplicates))
                i = j
            if unique and unique[0][0] != 0.0:
                unique.insert(0, (0.0, unique[0][1]))
            clips = []
            for idx in range(len(unique) - 1):
                start, vid = unique[idx]
                end, _ = unique[idx + 1]
                clips.append(mpy.VideoFileClip(vid).subclip(start, end))
            last_ts, last_vid = unique[-1]
            full = mpy.VideoFileClip(last_vid)
            if last_ts < full.duration:
                clips.append(full.subclip(last_ts, full.duration))
            return mpy.concatenate_videoclips(clips)

        mapping = {vp: ts for vp, ts in zip(video_paths, filtered)}
        final_clip = merge_videos(mapping)

        output_path = os.path.join(session_dir, "merged_output.mp4")
        final_clip.write_videofile(output_path, codec="libx264", fps=24, logger=None)

        return send_file(output_path, as_attachment=True)

    return render_template_string(HTML_FORM)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
