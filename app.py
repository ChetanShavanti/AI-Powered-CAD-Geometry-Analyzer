import streamlit as st
import streamlit.components.v1 as components
import os
from io import BytesIO
import requests
from stl import mesh
from streamlit_stl import stl_from_file

# --- Page Configuration ---
st.set_page_config(
    page_title="GeoMind AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Beautification ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Don't have a css file, so I will inject it directly
st.markdown("""
<style>
    body {
        color: #000000 !important; /* Black text globally */
    }
    /* Main app background */
    .stApp {
        background-color: #F0F8FF; /* Alice Blue */
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF; /* White */
    }
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    /* Metric styling */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.04);
    }
     /* Ensure metric label and value are visible */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #000000;
    }
    /* Insight card styling */
    .insight-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        background: #ffffff;
        color: #000000; /* Black text */
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.04);
        height: 100%;
    }
    .insight-card h4 {
        font-size: 1.1rem;
        margin-bottom: 10px;
        color: #5F9EA0; /* Cadet Blue */
    }
    .insight-card p {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    /* Tab styling */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 24px;
    }
body .stTabs [data-baseweb="tab"] {
    color: #000000;
}

    [data-testid="stTabs"] [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #000000;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background-color: #FFFFFF;
        color: #5F9EA0; /* Cadet Blue for active tab */
        border-bottom: 2px solid #5F9EA0;
    }
    /* Style for the file uploader button */
    [data-testid="stFileUploader"] button {
        background-color: #5F9EA0; /* Cadet Blue */
        color: #FFFFFF; /* White text */
        border: 1px solid #5F9EA0;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #538a8c; /* Darker Cadet Blue on hover */
        border: 1px solid #538a8c;
        color: #FFFFFF;
    }
    .stFileUploaderFile {
        background-color: #f0f2f6;
        border: 1px dashed #ced4da;
        border-radius: .25rem;
        padding: 1rem;
    }
    .stFileUploaderFileName {
        color: black;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #F0F8FF;
        color: black;
        text-align: right;
        padding: 10px;
        font-size: 12px;
    }
     h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stFileUploader, .stSelectbox, .stTextInput, .stTextArea {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="footer">Developed by Chetan Shavanti and Shreya Joshi</div>', unsafe_allow_html=True)


# --- OCC and STEP Analysis (Backend Logic) ---
try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    OCC_AVAILABLE = True
except ImportError:
    OCC_AVAILABLE = False
from steputils import p21

def generate_explanation(file_format, geometry_type, data):
    """
    Generate a comprehensive AI-based explanation using Hugging Face Inference Providers API.
    """
    api_token = os.environ.get("HUGGING_FACE_API_TOKEN")
    if not api_token:
        st.error("Hugging Face API token is not set. Please set the HUGGING_FACE_API_TOKEN environment variable.")
        return "API token not configured."


    stats_str = ""
    if file_format == 'STL':
        bbox = data.get('bbox', {})
        dims = [
            bbox['x'][1] - bbox['x'][0] if 'x' in bbox else 0,
            bbox['y'][1] - bbox['y'][0] if 'y' in bbox else 0,
            bbox['z'][1] - bbox['z'][0] if 'z' in bbox else 0
        ]
        stats_str = (
            f"- **Triangle Count:** {data.get('triangles', 'N/A')}\n"
            f"- **Bounding Box Dimensions (mm):** {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f}"
        )
    elif file_format == 'STEP':
        if 'faces' in data:
            stats_str = (
                f"- **Face Count:** {data.get('faces', 'N/A')}\n"
                f"- **Edge Count:** {data.get('edges', 'N/A')}"
            )
        else:
            stats_str = f"- **Entity Count:** {data.get('entities', 'N/A')}"

    prompt = f"""
    You are GeoMind AI, an expert assistant specializing in CAD geometry analysis. You have analyzed a CAD file with the following properties:
    - **File Format:** {file_format}
    - **Geometry Type:** {geometry_type}
    - **Key Statistics:**
    {stats_str}

    Based on this information, provide a detailed analysis for the user. Structure your response in well-written paragraphs under the following Markdown headings:

    ### 🤖 AI-Powered Analysis

    **1. Geometry Overview:**
    Describe what these statistics mean in practical terms (e.g., is the triangle count high or low? What does the bounding box tell us about the object's scale?).

    **2. Potential Use Cases:**
    List 3-5 specific and practical use cases. Be creative. For example, instead of just "3D printing," suggest "a housing for a Raspberry Pi project" or "a scale model for an architectural presentation."

    **3. Design & Manufacturing Insights:**
    - For **STL (Mesh)**: Discuss potential issues like mesh integrity (watertightness), non-manifold geometry, and how the triangle count (faceting) could impact the surface finish. Suggest software (like Blender or Meshmixer) for validation or repair.
    - For **STEP (B-Rep)**: Discuss the benefits of precise geometry for tasks like CNC machining, injection molding, or Finite Element Analysis (FEA). Mention the importance of model quality and continuity between surfaces.

    **4. Actionable Recommendations:**
    Provide a bulleted list of 2-3 concrete next steps for the user.
    - **Example for STL:** "Recommendation: Run a mesh analysis in a tool like Meshmixer to check for print-killing errors before sending to a 3D printer."
    - **Example for STEP:** "Recommendation: Import this model into a CAD assembly to verify its fit and interfaces with other parts."
    """

    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    data_payload = {"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 800, "temperature": 0.7}
    try:
        resp = requests.post("https://router.huggingface.co/v1/chat/completions", headers=headers, json=data_payload)
        if resp.status_code == 200:
            result = resp.json()
            if "choices" in result and result["choices"]:
                return result["choices"][0]["message"]["content"]
            else:
                st.warning(f"API returned an unexpected format: {result}")
                return "AI explanation could not be fully generated."
        else:
            st.error(f"API Error: {resp.status_code} - {resp.text}")
            return f"Failed to get AI explanation."
    except Exception as e:
        return f"Hugging Face API Error: {e}."

def detect_format(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    return {'step': 'STEP', 'stp': 'STEP', 'stl': 'STL'}.get(ext.strip('.'), None)

def analyze_step(file_path):
    if OCC_AVAILABLE:
        reader = STEPControl_Reader()
        if reader.ReadFile(file_path) != 1: raise ValueError("Failed to read STEP file")
        reader.TransferRoots()
        shape, faces, edges = reader.OneShape(), 0, 0
        explorer_face = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer_face.More(): faces += 1; explorer_face.Next()
        explorer_edge = TopExp_Explorer(shape, TopAbs_EDGE)
        while explorer_edge.More(): edges += 1; explorer_edge.Next()
        return {'faces': faces, 'edges': edges}
    else:
        with open(file_path, 'r') as f: data = p21.load(f)
        return {'entities': len(data)}

def analyze_stl(file_path, file_size):
    # Threshold: 50 MB
    LARGE_FILE_THRESHOLD = 50 * 1024 * 1024 

    stl_mesh = mesh.Mesh.from_file(file_path)
    min_c, max_c = stl_mesh.min_, stl_mesh.max_
    bbox = {'x': [float(min_c[0]), float(max_c[0])], 'y': [float(min_c[1]), float(max_c[1])], 'z': [float(min_c[2]), float(max_c[2])]}
    
    return {'triangles': stl_mesh.data.shape[0], 'bbox': bbox}


# --- UI: Sidebar ---
with st.sidebar:
    st.title("🤖 GeoMind AI")
    st.markdown("### AI-Powered CAD Geometry Analyzer")
    st.markdown("Welcome! Upload a CAD file to analyze its geometry, see a 3D preview, and get AI-powered insights.")
    
    uploaded_file = st.file_uploader("Choose a CAD file", type=['step', 'stp', 'stl'], help="Supported formats: STEP (.step/.stp), STL (.stl)")

    with open("testFiles.zip", "rb") as fp:
        st.download_button(
            label="Download Test Files",
            data=fp,
            file_name="testFiles.zip",
            mime="application/zip"
        )

    st.markdown("---")
    with st.expander("ℹ️ About CAD Formats"):
        st.markdown("""
        **B-Rep (Boundary Representation):** Used in STEP files. Represents geometry with exact mathematical surfaces.
        - **Pros:** High accuracy, ideal for manufacturing.
        - **Cons:** Larger files, complex to process.

        **Mesh (Triangle Mesh):** Used in STL files. Approximates surfaces with triangles.
        - **Pros:** Simple, great for 3D printing and visualization.
        - **Cons:** Approximate, not ideal for precise edits.
        """)
    with st.expander("❓ FAQ"):
        st.markdown("""
        **Q: Why is a STEP preview unavailable?**  
        A: Rendering STEP files requires heavy computational libraries, which are beyond the scope of this web-based POC.

        **Q: Is the AI explanation always perfect?**  
        A: It's a powerful guide, but for critical applications, always consult with a domain expert.
        """)

# --- UI: Main Content Area ---
if uploaded_file is None:
    st.markdown("## 👈 Get started by uploading a file")
    st.markdown("Once you upload a `.step`, `.stp`, or `.stl` file, the analysis will appear here.")
    st.image("https://www.svgrepo.com/show/508432/3d-printer.svg")

else:
    # Save uploaded file temporarily
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_format = detect_format(uploaded_file.name)

    # --- File Header ---
    st.markdown(f"### Analyzing: `{uploaded_file.name}`")
    size_mb = uploaded_file.size / 1024 / 1024
    st.caption(f"**File Size:** {size_mb:.2f} MB | **Format:** {file_format}")
    st.markdown("---")
    
    analysis_tab, preview_tab = st.tabs(["Analysis", "3D Preview"])

    with analysis_tab:
        data = None
        geometry_type = None

        try:
            if file_format == 'STEP':
                geometry_type = "B-Rep (Boundary Representation)"
                data = analyze_step(temp_file_path)
            elif file_format == 'STL':
                geometry_type = "Mesh (Triangle Mesh)"
                data = analyze_stl(temp_file_path, uploaded_file.size)

            # --- Insight Cards ---
            st.subheader("💡 At a Glance")
            cols = st.columns(3)
            cards_data = [
                ("Geometry Type", geometry_type.split('(')[0].strip()),
                ("Best Use Case", "Engineering" if file_format == 'STEP' else "3D Printing"),
                ("Accuracy", "High (Exact)" if file_format == 'STEP' else "Approximate")
            ]
            for i, (title, value) in enumerate(cards_data):
                with cols[i]:
                    st.markdown(f'<div class="insight-card"><h4>{title}</h4><p>{value}</p></div>', unsafe_allow_html=True)
            st.write("") # Spacer

            # --- AI Explanation Highlight Style ---
            st.markdown("""
            <style>
                .ai-explanation {
                    background-color: #E8F8F5;
                    border-left: 5px solid #1ABC9C;
                    padding: 20px;
                    border-radius: 5px;
                    margin-top: 20px;
                }
            </style>
            """, unsafe_allow_html=True)

            st.subheader("🤖 AI-Powered Explanation")
            with st.spinner("GeoMind AI is thinking..."):
                explanation = generate_explanation(file_format, geometry_type.split('(')[0], data)
            st.markdown(f'<div class="ai-explanation">{explanation}</div>', unsafe_allow_html=True)

            st.subheader("📊 Detailed Geometry Analysis")
            if file_format == 'STEP':
                st.metric("Geometry Type", "B-Rep (Boundary Representation)")
                if 'faces' in data:
                    col1, col2 = st.columns(2)
                    col1.metric("Face Count", data['faces'])
                    col2.metric("Edge Count", data['edges'])
                else:
                    st.metric("Entity Count", data['entities'])
                    st.info("Install `python-occ-core` for detailed face/edge counts.")
            elif file_format == 'STL':
                st.metric("Geometry Type", "Mesh (Triangle Mesh)")
                col1, col2 = st.columns(2)
                col1.metric("Triangle Count", data['triangles'])
                bbox = data['bbox']
                dims = f"{bbox['x'][1]-bbox['x'][0]:.1f} x {bbox['y'][1]-bbox['y'][0]:.1f} x {bbox['z'][1]-bbox['z'][0]:.1f} mm"
                col2.metric("Bounding Box", dims)
                st.write(f"**Min Coordinates (X,Y,Z):** {bbox['x'][0]:.2f}, {bbox['y'][0]:.2f}, {bbox['z'][0]:.2f}")
                st.write(f"**Max Coordinates (X,Y,Z):** {bbox['x'][1]:.2f}, {bbox['y'][1]:.2f}, {bbox['z'][1]:.2f}")

        except Exception as e:
            st.error(f"❌ An error occurred during analysis: {e}")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path) and 'preview_tab' not in locals():
                os.remove(temp_file_path)

    with preview_tab:
        if file_format == 'STL':
            st.subheader("🖼️ 3D Model Preview")
            stl_from_file(temp_file_path, height=600)
        else:
            st.info("3D preview is only available for STL files.")

    # Clean up the temporary file after both tabs are done
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)