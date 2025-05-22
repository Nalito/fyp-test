import streamlit as st
import os
import tempfile
from huggingface_hub import hf_hub_download
import moviepy.editor as mpy
import random

from extract_frames import VideoFrameExtractor
from get_inference import FramePredictor
from emo_trim import select_emotion

st.title("Emotion-Based Video Merger")

# Sidebar inputs
emo = st.sidebar.selectbox(
    "Select your preferred emotion:",
    ["sad", "happy", "fear", "surprise", "neutral", "disgust", "angry"]
)
videos = st.file_uploader(
    "Upload 2 to 4 video files:",
    type=["mp4", "avi", "mov", "mkv"],
    accept_multiple_files=True
)

if st.sidebar.button("Process Videos"):
    if not (2 <= len(videos) <= 4):
        st.error("Please upload between 2 and 4 video files.")
    else:
        # Create temporary session directory
        session_dir = tempfile.mkdtemp()
        video_paths = []
        for vid in videos:
            path = os.path.join(session_dir, vid.name)
            with open(path, "wb") as f:
                f.write(vid.getbuffer())
            video_paths.append(path)

        # 1. Frame extraction
        timestamps_list = []
        output_folders = []
        progress = st.progress(0)
        for idx, vp in enumerate(video_paths):
            extractor = VideoFrameExtractor(vp)
            extractor.extract_frames()
            timestamps_list.append(extractor.get_timestamps())
            output_folders.append(extractor.get_output_folder())
            progress.progress((idx + 1) / len(video_paths))

        # 2. Emotion inference
        predictions = []
        for out_folder in output_folders:
            model_path = hf_hub_download(repo_id="Nalito-dev/emo_detective", filename="best_model.h5")
            predictor = FramePredictor(model_path, out_folder)
            predictions.append(predictor.predict_frames())

        # 3. Select emotion timestamps
        filtered = []
        for preds, stamps in zip(predictions, timestamps_list):
            filtered.append(select_emotion(preds, stamps, emo))

        # 4. Merge videos
        def merge_videos(video_paths_timelines):
            # Build sorted unique timeline
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

        # 5. Export and display
        output_path = os.path.join(session_dir, "merged_output.mp4")
        final_clip.write_videofile(output_path, codec="libx264", fps=24, logger=None)
        st.video(output_path)
        st.success("Processing complete!")
