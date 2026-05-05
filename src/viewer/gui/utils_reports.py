import io

import numpy as np
import pandas as pd
import plotly.io as pio
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable, Image as RLImage, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)

from utils.utils_logger import setup_logger

logger = setup_logger()

REPORT_ROIS = [
    {'col': 'dlmuse-vol_601', 'label': 'Gray Matter'},
    {'col': 'dlmuse-vol_47',  'label': 'Hippocampus (R)'},
    {'col': 'dlmuse-vol_48',  'label': 'Hippocampus (L)'},
    {'col': 'dlmuse-vol_509', 'label': 'Ventricles'},
]


def _get_subject_row(df, mrid):
    rows = df[df['MRID'] == mrid]
    return rows.iloc[0] if not rows.empty else None


def _resolve_mrid(df):
    ss = st.session_state
    mrid = ss['user_plots']['selected'].get('mrid')
    if mrid is None or mrid not in df['MRID'].values:
        mrid = df['MRID'].iloc[0]
    return mrid


def _panel_view_report():
    ss = st.session_state
    df = ss['user_plots']['data'].get('df_user')
    if df is None:
        st.info('No user data available. Please prepare data in the **My Data** tab first.')
        return

    mrid = _resolve_mrid(df)
    row = _get_subject_row(df, mrid)
    if row is None:
        st.warning(f'No data for subject: {mrid}')
        return

    age = row.get('Age')
    sex = row.get('Sex')
    age_str = f'{float(age):.1f}' if pd.notna(age) else 'N/A'

    st.markdown('### NiChart Brain Aging Report')
    c1, c2, c3 = st.columns(3)
    c1.metric('Subject', str(mrid))
    c2.metric('Age', age_str)
    c3.metric('Sex', str(sex) if pd.notna(sex) else 'N/A')

    st.divider()
    st.markdown('#### Brain Region Centile Scores')
    st.caption('Centile scores show where the subject falls in the age-matched normative distribution.')

    for roi in REPORT_ROIS:
        col_name = roi['col']
        cent_col = col_name + '_cent'
        vol = row.get(col_name)
        centile = row.get(cent_col)

        v_ok = vol is not None and pd.notna(vol)
        c_ok = centile is not None and pd.notna(centile)

        with st.container(border=True):
            c1, c2, c3 = st.columns([2, 1, 3])
            with c1:
                st.markdown(f'**{roi["label"]}**')
                st.caption(f'{float(vol):,.0f} mm³' if v_ok else 'Volume: N/A')
            with c2:
                st.metric('Centile', f'{float(centile):.0f}' if c_ok else 'N/A')
            with c3:
                if c_ok:
                    cval = float(centile)
                    if cval < 5 or cval > 95:
                        st.caption(':red[Outlier (outside 5th–95th centile)]')
                    elif cval < 10 or cval > 90:
                        st.caption(':orange[Borderline]')
                    else:
                        st.caption(':green[Normal range]')
                    st.progress(cval / 100.0)
                else:
                    st.caption('No centile data available')


def _panel_report_settings():
    ss = st.session_state
    df = ss['user_plots']['data'].get('df_user')
    if df is None:
        st.info('No user data available yet. Please prepare data in the **My Data** tab first.')
        return

    with st.container(border=True):
        st.markdown('**Select Subject**')
        mrid_list = df['MRID'].tolist()
        curr = ss['user_plots']['selected'].get('mrid', mrid_list[0])
        idx = mrid_list.index(curr) if curr in mrid_list else 0
        sel = st.selectbox('Subject', mrid_list, index=idx, key='_rep_mrid_sel')
        ss['user_plots']['selected']['mrid'] = sel


def _generate_pdf(df, mrid):
    row = _get_subject_row(df, mrid)
    if row is None:
        return None

    age = row.get('Age')
    sex = row.get('Sex')
    age_str = f'{float(age):.1f}' if pd.notna(age) else 'N/A'
    sex_str = str(sex) if pd.notna(sex) else 'N/A'

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle('T',  parent=styles['Title'],   fontSize=18, spaceAfter=6)
    h2_s    = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13, spaceAfter=4)
    foot_s  = ParagraphStyle('F',  parent=styles['Normal'],  fontSize=9, textColor=colors.grey)

    story = [
        Paragraph('NiChart Brain Aging Report', title_s),
        HRFlowable(width='100%', thickness=1, color=colors.grey),
        Spacer(1, 0.15 * inch),
        Paragraph('Subject Information', h2_s),
    ]

    info_tbl = Table(
        [['Subject ID', str(mrid)], ['Age', age_str], ['Sex', sex_str]],
        colWidths=[2 * inch, 4 * inch],
    )
    info_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8e8e8')),
        ('FONTNAME',   (0, 0), (0, -1), 'Helvetica-Bold'),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING',    (0, 0), (-1, -1), 6),
    ]))
    story += [
        info_tbl,
        Spacer(1, 0.2 * inch),
        Paragraph('Brain Region Centile Scores', h2_s),
        Paragraph(
            "Centile scores indicate where a measurement falls in a normative distribution "
            "for the subject's age. Scores below 5 or above 95 are considered outliers.",
            styles['Normal'],
        ),
        Spacer(1, 0.1 * inch),
    ]

    tbl_data = [['Region', 'Volume (mm³)', 'Centile', 'Interpretation']]
    for roi in REPORT_ROIS:
        col_name = roi['col']
        cent_col = col_name + '_cent'
        vol = row.get(col_name)
        centile = row.get(cent_col)

        v_ok = vol is not None and pd.notna(vol)
        c_ok = centile is not None and pd.notna(centile)

        vol_str  = f'{float(vol):,.0f}' if v_ok else 'N/A'
        cent_str = f'{float(centile):.0f}' if c_ok else 'N/A'

        if not c_ok:
            interp = 'No data'
        elif float(centile) < 5 or float(centile) > 95:
            interp = 'Outlier'
        elif float(centile) < 10 or float(centile) > 90:
            interp = 'Borderline'
        else:
            interp = 'Normal range'

        tbl_data.append([roi['label'], vol_str, cent_str, interp])

    row_styles = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING',    (0, 0), (-1, -1), 6),
    ]
    for i in range(1, len(tbl_data)):
        bg = colors.white if i % 2 == 1 else colors.HexColor('#f5f5f5')
        row_styles.append(('BACKGROUND', (0, i), (-1, i), bg))

    roi_tbl = Table(tbl_data, colWidths=[2.2 * inch, 1.8 * inch, 1.3 * inch, 1.7 * inch])
    roi_tbl.setStyle(TableStyle(row_styles))
    story += [
        roi_tbl,
        Spacer(1, 0.3 * inch),
        HRFlowable(width='100%', thickness=0.5, color=colors.lightgrey),
        Spacer(1, 0.1 * inch),
        Paragraph('Generated by NiChart — Neuroimaging Chart of Age-Related Brain Changes', foot_s),
    ]

    doc.build(story)
    buf.seek(0)
    return buf.read()


def _panel_report_download():
    ss = st.session_state
    df = ss['user_plots']['data'].get('df_user')
    if df is None:
        st.info('No user data available. Please prepare data in the **My Data** tab first.')
        return

    mrid = _resolve_mrid(df)
    st.markdown(f'**Subject:** `{mrid}`')

    if st.button('Prepare PDF', key='_btn_prep_pdf', type='primary'):
        with st.spinner('Generating PDF...'):
            pdf_bytes = _generate_pdf(df, mrid)
        ss['_report_pdf_bytes'] = pdf_bytes
        ss['_report_pdf_mrid'] = mrid

    if ss.get('_report_pdf_bytes') is not None:
        st.download_button(
            label='Download PDF Report',
            data=ss['_report_pdf_bytes'],
            file_name=f'nichart_report_{ss["_report_pdf_mrid"]}.pdf',
            mime='application/pdf',
            key='_btn_dl_pdf',
        )


def _generate_plots_pdf(figs, ptype, title):
    """
    Render each cached plotly figure to PNG via kaleido and arrange them
    in the same grid layout (plots_per_row) used on screen.
    """
    ss = st.session_state

    n = len(figs)
    num_per_row = ss[ptype]['settings'].get('num_per_row', 1)
    flag_resize = ss[ptype]['settings'].get('flag_resize', True)
    plots_per_row = int(min(n, num_per_row)) if flag_resize else num_per_row
    plots_per_row = max(1, plots_per_row)

    page_w, page_h = letter
    margin = 0.5 * inch
    usable_w = page_w - 2 * margin
    img_w = usable_w / plots_per_row
    img_h = img_w * 0.65

    # Export figures to PNG at 2× resolution for sharpness
    px_w = int(img_w / inch * 144)
    px_h = int(img_h / inch * 144)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=margin, rightMargin=margin,
        topMargin=0.75 * inch, bottomMargin=0.5 * inch,
    )
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle('T', parent=styles['Title'], fontSize=16, spaceAfter=6)
    foot_s  = ParagraphStyle('F', parent=styles['Normal'], fontSize=9, textColor=colors.grey)

    story = [
        Paragraph(title, title_s),
        HRFlowable(width='100%', thickness=1, color=colors.grey),
        Spacer(1, 0.15 * inch),
    ]

    fig_list = list(figs.values())
    for row_start in range(0, n, plots_per_row):
        row_figs = fig_list[row_start:row_start + plots_per_row]
        row_cells = []
        for fig in row_figs:
            png = pio.to_image(fig, format='png', width=px_w, height=px_h, scale=1)
            row_cells.append(RLImage(io.BytesIO(png), width=img_w, height=img_h))
        while len(row_cells) < plots_per_row:
            row_cells.append('')

        tbl = Table([row_cells], colWidths=[img_w] * plots_per_row)
        tbl.setStyle(TableStyle([
            ('ALIGN',  (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.1 * inch))

    story += [
        HRFlowable(width='100%', thickness=0.5, color=colors.lightgrey),
        Spacer(1, 0.1 * inch),
        Paragraph('Generated by NiChart — Neuroimaging Chart of Age-Related Brain Changes', foot_s),
    ]

    doc.build(story)
    buf.seek(0)
    return buf.read()


def _generate_plots_html(figs, title):
    """Build a self-contained HTML report with interactive plotly charts."""
    plot_divs = [fig.to_html(include_plotlyjs=False, full_html=False) for fig in figs.values()]
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2em; color: #222; }}
    h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 0.3em; }}
    .plot-wrap {{ margin-bottom: 2em; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1em; }}
    footer {{ margin-top: 3em; font-size: 0.85em; color: #888; border-top: 1px solid #e0e0e0; padding-top: 0.5em; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {''.join(f'<div class="plot-wrap">{div}</div>' for div in plot_divs)}
  <footer>Generated by NiChart — Neuroimaging Chart of Age-Related Brain Changes</footer>
</body>
</html>"""


def panel_save_plots(ptype, title='NiChart Charts'):
    """
    Export current plots as PDF (same grid layout) or interactive HTML.
    Figures are cached in ss[ptype]['_figs'] when they are rendered.
    """
    ss = st.session_state
    figs = ss[ptype].get('_figs', {})

    if not figs:
        st.info('No plots to export yet. Add some plots in the **Plots** tab first.')
        return

    st.markdown(f'**{len(figs)} plot(s) ready to export.**')

    fmt = st.radio('Format', ['PDF', 'HTML'], horizontal=True, key=f'_rep_fmt_{ptype}')

    if st.button('Prepare', key=f'_btn_prep_plots_{ptype}', type='primary'):
        if fmt == 'PDF':
            with st.spinner('Rendering plots to PDF…'):
                ss[f'_plots_export_{ptype}'] = _generate_plots_pdf(figs, ptype, title)
        else:
            ss[f'_plots_export_{ptype}'] = _generate_plots_html(figs, title).encode()
        ss[f'_plots_export_fmt_{ptype}'] = fmt

    if ss.get(f'_plots_export_{ptype}') is not None:
        saved_fmt = ss[f'_plots_export_fmt_{ptype}']
        st.download_button(
            label=f'Download {saved_fmt}',
            data=ss[f'_plots_export_{ptype}'],
            file_name=f'nichart_plots.{"pdf" if saved_fmt == "PDF" else "html"}',
            mime='application/pdf' if saved_fmt == 'PDF' else 'text/html',
            key=f'_btn_dl_plots_{ptype}',
        )


def panel_reports():
    """Main reports panel with Report / Settings / Download sub-tabs."""
    rtab1, rtab2, rtab3 = st.tabs(['Report', 'Settings', 'Download'])

    with rtab1:
        _panel_view_report()

    with rtab2:
        _panel_report_settings()

    with rtab3:
        _panel_report_download()
