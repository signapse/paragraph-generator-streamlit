import streamlit as st
import os
import cv2
import subprocess
import pysrt
import copy
import csv
import pandas as pd
import tempfile
import zipfile
from io import BytesIO
import re

# Configure Streamlit page
st.set_page_config(
    page_title="SRT Processor",
    page_icon="🎬",
    layout="wide"
)


def srt_time_to_frames(sub_time, FPS):
    frame = 0
    frame += int((sub_time.milliseconds / 1000) * FPS)
    frame += sub_time.seconds * FPS
    frame += sub_time.minutes * FPS * 60
    frame += sub_time.hours * FPS * 60 * 60
    return frame


def frames_to_srt_time(frame, FPS):
    milliseconds = int(((frame % FPS) / FPS) * 1000)
    milliseconds_str = str(int(milliseconds))
    seconds = (frame / FPS)
    seconds_str = str(int(seconds % 60))
    minutes = seconds // 60
    minutes_str = str(int(minutes % 60))
    hours = minutes // 60
    hours_str = str(int(hours % 60))

    if len(milliseconds_str) == 1:
        milliseconds_str = f"00{milliseconds_str}"
    elif len(milliseconds_str) == 2:
        milliseconds_str = f"0{milliseconds_str}"

    if len(seconds_str) == 1:
        seconds_str = f"0{seconds_str}"

    if len(minutes_str) == 1:
        minutes_str = f"0{minutes_str}"

    if len(hours_str) == 1:
        hours_str = f"0{hours_str}"

    str_time = f'{hours_str}:{minutes_str}:{seconds_str},{milliseconds_str}'
    return str_time


def parse_sub_text(sub_text_in):
    sub_text_out = sub_text_in.strip().replace('\n', ' ').replace("&nbsp", "").replace(";", "")
    return sub_text_out


def parse_SRT_subs_to_sentences(subs, FPS):
    out_sentences = []
    all_text = ""
    buffer_text = ""
    buffer_start = None
    buffer_end = None
    buffer_indices = []

    def flush_buffer():
        nonlocal buffer_text, buffer_start, buffer_end, buffer_indices, all_text
        if buffer_text.strip():
            sentences = re.findall(r'[^.!?]+[.!?]', buffer_text)
            if not sentences:
                sentences = [buffer_text]
            total_chars = sum(len(s.strip()) for s in sentences)
            start_frame = srt_time_to_frames(buffer_start, FPS)
            end_frame = srt_time_to_frames(buffer_end, FPS)
            duration = end_frame - start_frame
            current_frame = start_frame
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if i < len(sentences) - 1:
                    sent_chars = len(sentence)
                    sent_duration = int(duration * (sent_chars / total_chars))
                    sent_end = current_frame + sent_duration
                else:
                    sent_end = end_frame
                out_sentences.append({
                    "text": sentence,
                    "start_frame": current_frame,
                    "end_frame": sent_end,
                    "seconds": (sent_end - current_frame) / FPS,
                    "text_id": sentence + "_".join(str(idx) for idx in buffer_indices)
                })
                all_text += sentence + " "
                current_frame = sent_end
        buffer_text = ""
        buffer_start = None
        buffer_end = None
        buffer_indices = []

    for idx, sub in enumerate(subs):
        sub_text = parse_sub_text(sub.text)
        if buffer_text == "":
            buffer_start = sub.start
        buffer_end = sub.end
        buffer_indices.append(idx)
        if buffer_text:
            buffer_text += " "
        buffer_text += sub_text
        if re.search(r'[.!?]$', sub_text):
            flush_buffer()

    if buffer_text.strip():
        flush_buffer()

    return out_sentences, all_text


def convert_sentences_to_paragraphs(out_sentences, request, FPS, paragraph_duration_threshold=15,
                                    wait_time_threshold=0.9, create_DCS=True):
    last_sentence = ""
    start_frame = 0
    paragraph_duration = 0
    sentences_in_paragraph = 0
    wait_time = 0
    paragraphs = []
    paragraphs_id = []
    paragraph = ""
    paragraph_id = ""
    para_n = 1

    para_sentences_data = []

    for s, sentence in enumerate(out_sentences):
        sentence_duration = sentence["seconds"]
        paragraph_duration_split = (paragraph_duration + sentence_duration) > paragraph_duration_threshold

        if s > 0:
            wait_time = (sentence["start_frame"] - last_sentence["end_frame"]) / FPS
        wait_time_split = wait_time > wait_time_threshold

        if (s > 0) and (paragraph_duration_split or wait_time_split):
            paragraph += f"({paragraph_duration:.2f})"
            paragraphs.append(paragraph)
            paragraphs_id.append(paragraph_id)

            paragraph = ""
            paragraph_id = ""
            start_frame = sentence["start_frame"]

            para_n += 1
            sentences_in_paragraph = 0
            paragraph_duration = 0

        paragraph += sentence["text"]
        paragraph_id += sentence["text_id"]
        paragraph += " "
        sentences_in_paragraph += 1
        paragraph_duration += sentence_duration

        end_frame = sentence["end_frame"]
        last_sentence = sentence

        para_sentences_data.append({
            'paragraph': para_n,
            'sentence': sentence["text"],
            'duration': sentence["seconds"]
        })

    paragraph += f"({paragraph_duration:.2f})"
    paragraphs.append(paragraph)
    paragraphs_id.append(paragraph_id)

    out_paragraphs = {}
    dcs_data = []
    ttg_data = []

    for p, paragraph_text in enumerate(paragraphs):
        str_p = str(p + 1)
        if len(str_p) == 1:
            str_p = f"0{str_p}"
        paragraph_label = f"{request}_001_{str_p}"

        if create_DCS:
            dcs_data.append({
                'Folder': request,
                'Video Label': paragraph_label,
                'Text': paragraph_text
            })
            ttg_data.append({
                'Sentence Label': paragraph_label,
                'Sentence Text': paragraph_text.split("(")[0].strip(),
                'BSL Gloss': ''
            })

        out_paragraphs[paragraph_label] = paragraphs_id[p]

    return out_paragraphs, para_sentences_data, dcs_data, ttg_data


def create_videos_SRT(out_sentences, paragraphs_id, FPS):
    new_subs = pysrt.SubRipFile()

    paragraphs = []
    for key in paragraphs_id.keys():
        paragraphs.append(f'{paragraphs_id[key]} \t {key}')

    i = 0
    if not paragraphs:
        return new_subs

    paragraph = paragraphs[i]
    video_title = paragraph.split("\t")[1]

    paragraph_text = video_title.strip()
    paragraph_index = 1
    paragraph_start = frames_to_srt_time(out_sentences[0]['start_frame'], FPS)
    paragraph_end = paragraph_start

    for out_sentence in out_sentences:
        sub_text = out_sentence["text_id"]

        if sub_text.replace(" ", "") in paragraph.split("\t")[0].replace(" ", ""):
            paragraph_end = frames_to_srt_time(out_sentence['end_frame'], FPS)
        else:
            # Add the current paragraph
            new_subs.append(
                pysrt.SubRipItem(paragraph_index, start=paragraph_start, end=paragraph_end, text=paragraph_text))
            i += 1

            # Move to next paragraph if available
            if i < len(paragraphs):
                paragraph = paragraphs[i]
                video_title = paragraph.split("\t")[1]
                paragraph_index += 1
                paragraph_text = video_title.strip()
                paragraph_start = frames_to_srt_time(out_sentence['start_frame'], FPS)
                paragraph_end = frames_to_srt_time(out_sentence['end_frame'], FPS)

    # Add the final paragraph
    if paragraph_text:
        new_subs.append(
            pysrt.SubRipItem(paragraph_index, start=paragraph_start, end=paragraph_end, text=paragraph_text))

    return new_subs


def main():
    st.title("🎬 SRT Processor")
    st.write("Upload your SRT and MP4 files to process subtitles into paragraphs for translation workflows.")

    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'results_data' not in st.session_state:
        st.session_state.results_data = None

    # Create two columns for file uploads
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Upload SRT File")
        srt_file = st.file_uploader("Choose an SRT file", type=['srt'], key="srt_uploader")

    with col2:
        st.subheader("🎥 Upload MP4 File")
        mp4_file = st.file_uploader("Choose an MP4 file", type=['mp4'], key="mp4_uploader")

    # Configuration options
    st.subheader("⚙️ Configuration")

    # Auto-detect request name from uploaded files
    auto_request_name = ""
    if srt_file is not None:
        # Remove .srt extension and use as request name
        auto_request_name = srt_file.name.rsplit('.', 1)[0]
    elif mp4_file is not None:
        # Remove .mp4 extension and use as request name
        auto_request_name = mp4_file.name.rsplit('.', 1)[0]

    request_name = st.text_input(
        "Request Name",
        value=auto_request_name if auto_request_name else "sample_project",
        help="This will be used as the prefix for output files. Auto-detected from uploaded file names."
    )

    # Fixed thresholds as per original script
    paragraph_duration_threshold = 1000
    wait_time_threshold = 1000

    # Reset processed state if files change
    if srt_file is not None or mp4_file is not None:
        # Check if files are different from last processing
        current_files = (srt_file.name if srt_file else None, mp4_file.name if mp4_file else None)
        if 'last_files' not in st.session_state:
            st.session_state.last_files = (None, None)

        if current_files != st.session_state.last_files:
            st.session_state.processed = False
            st.session_state.results_data = None
            st.session_state.last_files = current_files

    if st.button("🚀 Process Files", type="primary", disabled=st.session_state.processed):
        if srt_file is None:
            st.error("Please upload an SRT file!")
            return

        if mp4_file is None:
            st.error("Please upload an MP4 file!")
            return

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Save uploaded files to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save SRT file
                srt_path = os.path.join(temp_dir, "input.srt")
                with open(srt_path, "wb") as f:
                    f.write(srt_file.getbuffer())

                # Save MP4 file
                mp4_path = os.path.join(temp_dir, "input.mp4")
                with open(mp4_path, "wb") as f:
                    f.write(mp4_file.getbuffer())

                progress_bar.progress(20)
                status_text.text("📹 Analyzing video properties...")

                # Get FPS from video
                input_vidcap = cv2.VideoCapture(mp4_path)
                FPS = round(input_vidcap.get(cv2.CAP_PROP_FPS))
                if FPS == 0:
                    FPS = 25
                input_vidcap.release()

                progress_bar.progress(40)
                status_text.text("📝 Processing SRT subtitles...")

                # Process SRT
                subs = pysrt.open(srt_path)
                out_sentences, all_text = parse_SRT_subs_to_sentences(subs, FPS)

                progress_bar.progress(60)
                status_text.text("📋 Converting to paragraphs...")

                # Convert to paragraphs
                out_paragraphs, para_sentences_data, dcs_data, ttg_data = convert_sentences_to_paragraphs(
                    out_sentences=out_sentences,
                    request=request_name,
                    FPS=FPS,
                    paragraph_duration_threshold=paragraph_duration_threshold,
                    wait_time_threshold=wait_time_threshold
                )

                progress_bar.progress(80)
                status_text.text("🎬 Creating video SRT...")

                # Create video SRT
                out_paragraphs_video = convert_sentences_to_paragraphs(
                    out_sentences=out_sentences,
                    request=request_name,
                    FPS=FPS,
                    paragraph_duration_threshold=0,
                    wait_time_threshold=0,
                    create_DCS=False
                )[0]

                video_srt = create_videos_SRT(out_sentences, out_paragraphs_video, FPS)

                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")

                # Display results
                st.success(
                    f"Successfully processed {len(out_sentences)} sentences into {len(out_paragraphs)} paragraphs!")

                # Show some statistics
                col6, col7, col8 = st.columns(3)
                with col6:
                    st.metric("Total Sentences", len(out_sentences))
                with col7:
                    st.metric("Total Paragraphs", len(out_paragraphs))
                with col8:
                    st.metric("Video FPS", FPS)

                # Create download files
                st.subheader("📥 Download Results")

                # Create a zip file with only the required outputs
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:

                    # DCS Paragraphs CSV
                    if dcs_data:
                        dcs_df = pd.DataFrame(dcs_data)
                        csv_content = dcs_df.to_csv(index=False)
                        zip_file.writestr(f"{request_name}_DCS_Paragraphs.csv", csv_content)

                    # Video SRT (the main output) - properly format the SRT content
                    video_srt_content = ""
                    for item in video_srt:
                        video_srt_content += f"{item.index}\n"
                        video_srt_content += f"{item.start} --> {item.end}\n"
                        video_srt_content += f"{item.text}\n\n"
                    zip_file.writestr(f"{request_name}_Videos.srt", video_srt_content)

                # Store results in session state
                st.session_state.results_data = {
                    'zip_buffer': zip_buffer.getvalue(),
                    'request_name': request_name,
                    'out_sentences': out_sentences,
                    'out_paragraphs': out_paragraphs,
                    'dcs_data': dcs_data,
                    'video_srt': video_srt,
                    'FPS': FPS
                }
                st.session_state.processed = True

                # Success message and metrics
                st.success(
                    f"Successfully processed {len(out_sentences)} sentences into {len(out_paragraphs)} paragraphs!")

                # Show some statistics
                col6, col7, col8 = st.columns(3)
                with col6:
                    st.metric("Total Sentences", len(out_sentences))
                with col7:
                    st.metric("Total Paragraphs", len(out_paragraphs))
                with col8:
                    st.metric("Video FPS", FPS)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your file formats and try again.")

    # Display results if processing is complete
    if st.session_state.processed and st.session_state.results_data:
        results = st.session_state.results_data

        # Create download section
        st.subheader("📥 Download Results")

        col_download, col_reset = st.columns([3, 1])

        with col_download:
            st.download_button(
                label="📦 Download Results (ZIP)",
                data=results['zip_buffer'],
                file_name=f"{results['request_name']}_processed_results.zip",
                mime="application/zip"
            )

        with col_reset:
            if st.button("🔄 Process New Files", type="secondary"):
                st.session_state.processed = False
                st.session_state.results_data = None
                st.rerun()

        # Show preview of results
        if st.checkbox("👀 Show Preview of Results"):
            st.subheader("Sample Sentences")
            df_sentences = pd.DataFrame(results['out_sentences'][:5])  # Show first 5
            st.dataframe(df_sentences[['text', 'seconds']])

            if results['dcs_data']:
                st.subheader("Sample Paragraphs")
                df_paragraphs = pd.DataFrame(results['dcs_data'][:3])  # Show first 3
                st.dataframe(df_paragraphs)

            st.subheader("Video SRT Preview")
            # Properly format SRT for preview
            srt_preview = ""
            for i, item in enumerate(results['video_srt'][:3]):  # Show first 3 items
                srt_preview += f"{item.index}\n{item.start} --> {item.end}\n{item.text}\n\n"
            st.text(srt_preview)


if __name__ == "__main__":
    main()