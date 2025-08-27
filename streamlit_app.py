import streamlit as st
import os
import pysrt
import copy
import csv
import pandas as pd
import tempfile
import zipfile
from io import BytesIO
import re
from datetime import timedelta

# Configure Streamlit page
st.set_page_config(
    page_title="SRT Processor",
    page_icon="🎬",
    layout="wide"
)


def srt_time_to_seconds(sub_time):
    """Convert SRT time to total seconds (float)"""
    total_seconds = 0
    total_seconds += sub_time.milliseconds / 1000
    total_seconds += sub_time.seconds
    total_seconds += sub_time.minutes * 60
    total_seconds += sub_time.hours * 60 * 60
    return total_seconds


def seconds_to_srt_time(seconds):
    """Convert seconds back to SRT time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def parse_sub_text(sub_text_in):
    sub_text_out = sub_text_in.strip().replace('\n', ' ').replace("&nbsp", "").replace(";", "")
    return sub_text_out


def parse_SRT_subs_to_sentences(subs):
    """Parse SRT subtitles into sentences with timing information"""
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
            start_seconds = srt_time_to_seconds(buffer_start)
            end_seconds = srt_time_to_seconds(buffer_end)
            duration = end_seconds - start_seconds
            current_seconds = start_seconds

            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue

                if i < len(sentences) - 1:
                    sent_chars = len(sentence)
                    sent_duration = duration * (sent_chars / total_chars)
                    sent_end = current_seconds + sent_duration
                else:
                    sent_end = end_seconds

                out_sentences.append({
                    "text": sentence,
                    "start_seconds": current_seconds,
                    "end_seconds": sent_end,
                    "duration": sent_end - current_seconds,
                    "text_id": sentence + "_".join(str(idx) for idx in buffer_indices)
                })
                all_text += sentence + " "
                current_seconds = sent_end

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


def convert_sentences_to_paragraphs(out_sentences, request, paragraph_duration_threshold=15,
                                    wait_time_threshold=0.9, create_DCS=True):
    """Convert sentences to paragraphs based on duration and wait time thresholds"""
    last_sentence = ""
    start_seconds = 0
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
        sentence_duration = sentence["duration"]
        paragraph_duration_split = (paragraph_duration + sentence_duration) > paragraph_duration_threshold

        if s > 0:
            wait_time = sentence["start_seconds"] - last_sentence["end_seconds"]
        wait_time_split = wait_time > wait_time_threshold

        if (s > 0) and (paragraph_duration_split or wait_time_split):
            paragraph += f"({paragraph_duration:.2f})"
            paragraphs.append(paragraph)
            paragraphs_id.append(paragraph_id)

            paragraph = ""
            paragraph_id = ""
            start_seconds = sentence["start_seconds"]

            para_n += 1
            sentences_in_paragraph = 0
            paragraph_duration = 0

        paragraph += sentence["text"]
        paragraph_id += sentence["text_id"]
        paragraph += " "
        sentences_in_paragraph += 1
        paragraph_duration += sentence_duration

        end_seconds = sentence["end_seconds"]
        last_sentence = sentence

        para_sentences_data.append({
            'paragraph': para_n,
            'sentence': sentence["text"],
            'duration': sentence["duration"]
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


def create_videos_SRT(out_sentences, paragraphs_id):
    """Create new SRT file with paragraph-based timing"""
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
    paragraph_start_seconds = out_sentences[0]['start_seconds']
    paragraph_end_seconds = paragraph_start_seconds

    for out_sentence in out_sentences:
        sub_text = out_sentence["text_id"]

        if sub_text.replace(" ", "") in paragraph.split("\t")[0].replace(" ", ""):
            paragraph_end_seconds = out_sentence['end_seconds']
        else:
            # Add the current paragraph
            start_time = pysrt.SubRipTime.from_ordinal(int(paragraph_start_seconds * 1000))
            end_time = pysrt.SubRipTime.from_ordinal(int(paragraph_end_seconds * 1000))
            new_subs.append(
                pysrt.SubRipItem(paragraph_index, start=start_time, end=end_time, text=paragraph_text))
            i += 1

            # Move to next paragraph if available
            if i < len(paragraphs):
                paragraph = paragraphs[i]
                video_title = paragraph.split("\t")[1]
                paragraph_index += 1
                paragraph_text = video_title.strip()
                paragraph_start_seconds = out_sentence['start_seconds']
                paragraph_end_seconds = out_sentence['end_seconds']

    # Add the final paragraph
    if paragraph_text:
        start_time = pysrt.SubRipTime.from_ordinal(int(paragraph_start_seconds * 1000))
        end_time = pysrt.SubRipTime.from_ordinal(int(paragraph_end_seconds * 1000))
        new_subs.append(
            pysrt.SubRipItem(paragraph_index, start=start_time, end=end_time, text=paragraph_text))

    return new_subs


def main():
    st.title("🎬 SRT Processor")
    st.write("Upload your SRT file to process subtitles into paragraphs for translation workflows.")

    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'results_data' not in st.session_state:
        st.session_state.results_data = None

    # File upload section
    st.subheader("📄 Upload SRT File")
    srt_file = st.file_uploader("Choose an SRT file", type=['srt'], key="srt_uploader")

    # Configuration options
    st.subheader("⚙️ Configuration")

    # Auto-detect request name from uploaded file
    auto_request_name = ""
    if srt_file is not None:
        # Remove .srt extension and use as request name
        auto_request_name = srt_file.name.rsplit('.', 1)[0]

    request_name = st.text_input(
        "Request Name",
        value=auto_request_name if auto_request_name else "sample_project",
        help="This will be used as the prefix for output files. Auto-detected from uploaded file name."
    )

    # Configurable thresholds
    col1, col2 = st.columns(2)
    with col1:
        paragraph_duration_threshold = st.number_input(
            "Paragraph Duration Threshold (seconds)",
            min_value=5.0,
            max_value=60.0,
            value=15.0,
            step=1.0,
            help="Maximum duration for a paragraph before splitting"
        )

    with col2:
        wait_time_threshold = st.number_input(
            "Wait Time Threshold (seconds)",
            min_value=0.1,
            max_value=5.0,
            value=0.9,
            step=0.1,
            help="Minimum pause duration to trigger paragraph split"
        )

    # Reset processed state if file changes
    if srt_file is not None:
        # Check if file is different from last processing
        current_file = srt_file.name if srt_file else None
        if 'last_file' not in st.session_state:
            st.session_state.last_file = None

        if current_file != st.session_state.last_file:
            st.session_state.processed = False
            st.session_state.results_data = None
            st.session_state.last_file = current_file

    if st.button("🚀 Process SRT File", type="primary", disabled=st.session_state.processed):
        if srt_file is None:
            st.error("Please upload an SRT file!")
            return

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Save uploaded file to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save SRT file
                srt_path = os.path.join(temp_dir, "input.srt")
                with open(srt_path, "wb") as f:
                    f.write(srt_file.getbuffer())

                progress_bar.progress(25)
                status_text.text("📝 Processing SRT subtitles...")

                # Process SRT
                subs = pysrt.open(srt_path)
                out_sentences, all_text = parse_SRT_subs_to_sentences(subs)

                progress_bar.progress(50)
                status_text.text("📋 Converting to paragraphs...")

                # Convert to paragraphs
                out_paragraphs, para_sentences_data, dcs_data, ttg_data = convert_sentences_to_paragraphs(
                    out_sentences=out_sentences,
                    request=request_name,
                    paragraph_duration_threshold=paragraph_duration_threshold,
                    wait_time_threshold=wait_time_threshold
                )

                progress_bar.progress(75)
                status_text.text("🎬 Creating video SRT...")

                # Create video SRT (no splits for continuous playback)
                out_paragraphs_video = convert_sentences_to_paragraphs(
                    out_sentences=out_sentences,
                    request=request_name,
                    paragraph_duration_threshold=float('inf'),  # No duration splits
                    wait_time_threshold=float('inf'),  # No wait time splits
                    create_DCS=False
                )[0]

                video_srt = create_videos_SRT(out_sentences, out_paragraphs_video)

                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")

                # Store results in session state
                st.session_state.results_data = {
                    'request_name': request_name,
                    'out_sentences': out_sentences,
                    'out_paragraphs': out_paragraphs,
                    'dcs_data': dcs_data,
                    'video_srt': video_srt,
                    'total_duration': sum(s['duration'] for s in out_sentences)
                }
                st.session_state.processed = True

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your SRT file format and try again.")

    # Display results if processing is complete
    if st.session_state.processed and st.session_state.results_data:
        results = st.session_state.results_data

        # Success message and metrics
        st.success(
            f"Successfully processed {len(results['out_sentences'])} sentences into {len(results['out_paragraphs'])} paragraphs!"
        )

        # Show statistics
        col6, col7, col8 = st.columns(3)
        with col6:
            st.metric("Total Sentences", len(results['out_sentences']))
        with col7:
            st.metric("Total Paragraphs", len(results['out_paragraphs']))
        with col8:
            st.metric("Total Duration", f"{results['total_duration']:.1f}s")

        # Create download section
        st.subheader("📥 Download Results")

        # Create download files
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # DCS Paragraphs CSV
            if results['dcs_data']:
                dcs_df = pd.DataFrame(results['dcs_data'])
                csv_content = dcs_df.to_csv(index=False)
                zip_file.writestr(f"{results['request_name']}_DCS_Paragraphs.csv", csv_content)

            # Video SRT (the main output)
            video_srt_content = ""
            for item in results['video_srt']:
                video_srt_content += f"{item.index}\n"
                video_srt_content += f"{item.start} --> {item.end}\n"
                video_srt_content += f"{item.text}\n\n"
            zip_file.writestr(f"{results['request_name']}_Videos.srt", video_srt_content)

        col_download, col_reset = st.columns([3, 1])

        with col_download:
            st.download_button(
                label="📦 Download Results (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"{results['request_name']}_processed_results.zip",
                mime="application/zip"
            )

        with col_reset:
            if st.button("🔄 Process New File", type="secondary"):
                st.session_state.processed = False
                st.session_state.results_data = None
                st.rerun()

        # Show preview of results
        if st.checkbox("👀 Show Preview of Results"):
            st.subheader("Sample Sentences")
            df_sentences = pd.DataFrame(results['out_sentences'][:5])  # Show first 5
            st.dataframe(df_sentences[['text', 'duration']].round(2))

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