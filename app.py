"""Streamlit interface for worker tracking and PPE review."""
from datetime import datetime
from pathlib import Path
import uuid

import streamlit as st
import torch

from monitor import run
from download_models import ensure_models

PPE_LABELS = {
    'helmet': 'Helmet',
    'vest': 'Vest',
    'gloves': 'Gloves (advisory)',
    'boots': 'Boots (advisory)',
}
SOURCE_LABELS = {
    'model': 'Model',
    'model+colour': 'Model + colour assist',
    'colour': 'Colour assist only',
}


def format_status(value):
    """Format a tri-state PPE observation for display."""
    return 'Checking' if value is None else 'Seen' if value else 'Not seen'


def worker_table(rows):
    """Build readable table rows from the pipeline observations."""
    return [
        {
            'Worker': row['worker'],
            'Activity': row['activity'].replace('_', ' ').title(),
            **{PPE_LABELS[item]: format_status(value)
               for item, value in row['ppe'].items()},
            'Current vest evidence': SOURCE_LABELS.get(
                row['ppe_sources'].get('vest'), 'No current detection'),
        }
        for row in rows
    ]


st.set_page_config(page_title='Worker Monitor', layout='wide')
st.title('Worker Monitor')
st.write('Track workers, review protective equipment, and identify walking, standing, or sitting.')
st.caption('Helmet and vest: Seen / Not seen / Checking. Not seen does not prove an item is absent. Gloves and boots are advisory.')

upload = st.file_uploader('Upload a video', type=['mp4', 'avi', 'mov', 'mkv'])
path = st.text_input('Or enter a local video path')
stride = st.slider('Analyse one frame every N frames', 1, 10, 2,
                   help='Lower values provide more frequent observations; higher values reduce processing time.')

gpu_ready = torch.cuda.is_available()
device = st.selectbox(
    'Processing device', ['0', 'auto', 'cpu'] if gpu_ready else ['cpu', 'auto'],
    format_func=lambda value: {'0': 'NVIDIA GPU', 'auto': 'Automatic', 'cpu': 'CPU'}[value],
)
if gpu_ready:
    st.success('GPU acceleration is available.')
else:
    st.info('GPU acceleration is unavailable in the current environment.')
st.caption('For webcam input, use the command-line preview and press Q to stop.')

colour_assist = st.checkbox('Fluorescent colour assist for open vests', value=True)
st.caption('Looks for two fluorescent bands on the torso. The table shows the evidence source; similar ordinary clothing can produce false detections.')
with st.expander('Model setup'):
    st.write('Required models download automatically on first use. Download them in advance for offline use.')
    st.caption('The complete model cache uses about 631 MiB. Downloads are verified against SHA-256 checksums.')
    if st.button('Download / verify models'):
        progress = st.empty()
        try:
            with st.spinner('Preparing model files...'):
                ensure_models(progress=progress.info)
            st.success('All models are ready.')
        except Exception as error:
            st.error(str(error))
image = st.empty()
table = st.empty()

if st.button('Start analysis', disabled=not (upload or path)):
    folder = Path(__file__).parent / 'output' / (
        datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
    folder.mkdir(parents=True)
    source = path
    if upload:
        source = str(folder / ('input' + Path(upload.name).suffix.lower()))
        with open(source, 'wb') as target:
            target.write(upload.getbuffer())

    def show(frame, rows):
        image.image(frame, channels='BGR', width='stretch')
        table.dataframe(worker_table(rows))

    try:
        with st.spinner('Analysing video...'):
            result = run(source, folder, stride, None if device == 'auto' else device,
                         on_frame=show, colour_assist=colour_assist)
        st.success(f'Analysis complete. Results saved to {folder.name}.')
        st.json(result)
        for name, mime in [('annotated.mp4', 'video/mp4'),
                           ('workers.jsonl', 'application/x-ndjson'),
                           ('summary.json', 'application/json')]:
            file = folder / name
            if file.exists():
                st.download_button(name, file.read_bytes(), file_name=name, mime=mime)
    except Exception as error:
        st.error(str(error))
